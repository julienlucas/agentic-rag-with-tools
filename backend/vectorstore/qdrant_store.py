"""
Qdrant multi-tenant : chaque utilisateur a son espace, isolé par `tenant_id`.

Modèle recommandé par Qdrant (plutôt qu'une collection par utilisateur, qui ne tient pas
à l'échelle) :
- une collection `chunks` partagée ; chaque point porte `tenant_id` dans son payload ;
- index de payload `tenant_id` déclaré `is_tenant` : les points d'un tenant sont rangés
  ensemble sur disque ;
- HNSW `m=0` (pas de graphe global) + `payload_m` : Qdrant construit un graphe HNSW PAR
  tenant. Une recherche filtrée sur un tenant ne parcourt que ses vecteurs.

Deux vecteurs par chunk, recherche hybride côté serveur :
- `dense`  : Cohere Embed v4 (Bedrock), cosinus ;
- `bm25`   : vecteur sparse BM25 (fastembed « Qdrant/bm25 »), IDF calculé par Qdrant
             (modifier IDF) ;
fusionnés par RRF dans une seule requête (`query_points` + prefetch).

La collection `pages` (sans vecteurs) garde les pages OCR pour grep / read_page, et un point
`kind="document"` par fichier : le registre des documents du tenant. Ce point est écrit EN
DERNIER : un document n'existe qu'une fois tous ses chunks et pages écrits.

SÉCURITÉ : ce module est le seul à parler à Qdrant. Tout accès passe par TenantStore, qui
ajoute le filtre `tenant_id` à chaque lecture, recherche et suppression. Ne jamais exposer
le client hors d'ici.
"""
import re
import threading
import time
import uuid
from typing import Dict, Iterable, List, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from ..config.settings import settings
from ..utils.logging import logger

# Espace de noms des identifiants de points : uuid5(tenant, fichier, rang) rend l'ingestion
# idempotente (réécrire un fichier remplace ses points au lieu de les dupliquer).
_POINT_NS = uuid.UUID("6f1c2b8e-4a57-4d0e-9a51-3c1d2f7e8b90")
_TENANT_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
_UPSERT_BATCH = 64

_DISTANCES = {"cosine": models.Distance.COSINE, "l2": models.Distance.EUCLID, "ip": models.Distance.DOT}


class QuotaExceeded(Exception):
    """Le tenant dépasserait son quota de documents ou de chunks."""


class Bm25Encoder:
    """Vecteurs sparse BM25 (fastembed). Le modèle (~stopwords + stemmer) est chargé au premier usage."""

    def __init__(self, language: str = None):
        self.language = language or settings.BM25_LANGUAGE
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from fastembed import SparseTextEmbedding
                self._model = SparseTextEmbedding("Qdrant/bm25", language=self.language)
        return self._model

    @staticmethod
    def _to_qdrant(v) -> models.SparseVector:
        return models.SparseVector(indices=[int(i) for i in v.indices], values=[float(x) for x in v.values])

    def embed_documents(self, texts: List[str]) -> List[models.SparseVector]:
        return [self._to_qdrant(v) for v in self._get().embed(texts)]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._to_qdrant(next(iter(self._get().query_embed(text))))


def _batches(items: List, size: int) -> Iterable[List]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


class QdrantStore:
    """Connexion + schéma des collections. Ne s'utilise qu'à travers for_tenant()."""

    def __init__(self, client: QdrantClient = None, embeddings=None, sparse: Bm25Encoder = None):
        self._client = client or self._connect()
        self._embeddings = embeddings
        self.sparse = sparse or Bm25Encoder()
        self.chunks = settings.QDRANT_CHUNKS_COLLECTION
        self.pages = settings.QDRANT_PAGES_COLLECTION
        self._ensure_schema()

    @staticmethod
    def _connect() -> QdrantClient:
        if settings.QDRANT_URL:
            return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=30)
        logger.warning("QDRANT_URL absente : Qdrant en mémoire, les données sont perdues au redémarrage.")
        return QdrantClient(":memory:")

    @property
    def embeddings(self):
        if self._embeddings is None:
            from ..retriever.embeddings import get_embeddings
            self._embeddings = get_embeddings()
        return self._embeddings

    def _tenant_index(self, collection: str) -> None:
        self._client.create_payload_index(
            collection, "tenant_id",
            field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
        )

    def _keyword_index(self, collection: str, field: str) -> None:
        self._client.create_payload_index(collection, field, field_schema=models.PayloadSchemaType.KEYWORD)

    def _ensure_schema(self) -> None:
        if not self._client.collection_exists(self.chunks):
            self._client.create_collection(
                self.chunks,
                vectors_config={"dense": models.VectorParams(
                    size=settings.EMBEDDING_DIMENSIONS, distance=_DISTANCES[settings.VECTOR_SPACE],
                )},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
                hnsw_config=models.HnswConfigDiff(
                    m=0, payload_m=settings.QDRANT_HNSW_PAYLOAD_M, ef_construct=settings.QDRANT_HNSW_EF_CONSTRUCT,
                ),
                # int8 : ~4x moins de RAM pour les vecteurs, rescoring sur les originaux.
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True),
                ),
            )
            # L'index tenant doit exister AVANT l'indexation pour que les graphes par tenant se construisent.
            self._tenant_index(self.chunks)
            self._keyword_index(self.chunks, "file_hash")
            self._keyword_index(self.chunks, "source")
            logger.info(f"Collection Qdrant créée: {self.chunks}")
        else:
            dense = self._client.get_collection(self.chunks).config.params.vectors["dense"]
            if dense.size != settings.EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    f"Collection {self.chunks}: vecteurs de dimension {dense.size}, "
                    f"EMBEDDING_DIMENSIONS={settings.EMBEDDING_DIMENSIONS}. Recréer la collection."
                )

        if not self._client.collection_exists(self.pages):
            self._client.create_collection(self.pages, vectors_config={})
            self._tenant_index(self.pages)
            for field in ("kind", "file_hash"):
                self._keyword_index(self.pages, field)
            logger.info(f"Collection Qdrant créée: {self.pages}")

    def for_tenant(self, tenant_id: str) -> "TenantStore":
        return TenantStore(self, self._client, tenant_id)


class TenantStore:
    """L'espace d'un utilisateur. Chaque opération est filtrée sur son tenant_id."""

    def __init__(self, store: QdrantStore, client: QdrantClient, tenant_id: str):
        if not isinstance(tenant_id, str) or not _TENANT_RE.match(tenant_id):
            raise ValueError(f"tenant_id invalide: {tenant_id!r}")
        self._store = store
        self._client = client
        self.tenant_id = tenant_id

    # --- filtres et identifiants ---------------------------------------------

    def _filter(self, *conditions: models.Condition) -> models.Filter:
        return models.Filter(must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=self.tenant_id)),
            *conditions,
        ])

    @staticmethod
    def _eq(key: str, value) -> models.FieldCondition:
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    def _point_id(self, file_hash: str, kind: str, rank: int = 0) -> str:
        return str(uuid.uuid5(_POINT_NS, f"{self.tenant_id}:{file_hash}:{kind}:{rank}"))

    def _scroll(self, collection: str, flt: models.Filter) -> List[models.Record]:
        records, offset = [], None
        while True:
            batch, offset = self._client.scroll(
                collection, scroll_filter=flt, limit=256, offset=offset, with_payload=True, with_vectors=False,
            )
            records.extend(batch)
            if offset is None:
                return records

    # --- registre des documents ----------------------------------------------

    def documents(self) -> List[Dict]:
        """Documents complètement indexés : [{file_hash, source, chunk_count, page_count, created_at}]."""
        records = self._scroll(self._store.pages, self._filter(self._eq("kind", "document")))
        docs = [{k: v for k, v in r.payload.items() if k not in ("tenant_id", "kind")} for r in records]
        return sorted(docs, key=lambda d: d.get("created_at", 0))

    def sources(self) -> List[str]:
        return [d["source"] for d in self.documents()]

    def has_document(self, file_hash: str) -> bool:
        flt = self._filter(self._eq("kind", "document"), self._eq("file_hash", file_hash))
        return self._client.count(self._store.pages, count_filter=flt, exact=True).count > 0

    def chunk_count(self) -> int:
        return self._client.count(self._store.chunks, count_filter=self._filter(), exact=True).count

    # --- écriture -------------------------------------------------------------

    def add_document(self, file_hash: str, source: str, chunks: List[Document], pages: List[str],
                     enforce_quota: bool = True) -> int:
        """
        Indexe un fichier (chunks + pages). Idempotent : un fichier déjà présent (même hash) est
        réécrit à l'identique ; un fichier de même nom mais de contenu différent remplace l'ancien.
        Retourne le nombre de chunks indexés. `enforce_quota=False` : réservé à l'évaluation.
        """
        known = self.documents()
        replaced = [d for d in known if d["file_hash"] == file_hash or d["source"] == source]
        if enforce_quota:
            if len(known) - len(replaced) + 1 > settings.TENANT_MAX_DOCUMENTS:
                raise QuotaExceeded(f"Quota atteint : {settings.TENANT_MAX_DOCUMENTS} documents par espace.")
            freed = sum(d.get("chunk_count", 0) for d in replaced)
            if self.chunk_count() - freed + len(chunks) > settings.TENANT_MAX_CHUNKS:
                raise QuotaExceeded(f"Quota atteint : {settings.TENANT_MAX_CHUNKS} passages indexés par espace.")

        # Purge d'abord : restes d'une indexation interrompue, ou ancienne version du fichier.
        self.delete_document(file_hash)
        for d in replaced:
            if d["file_hash"] != file_hash:
                self.delete_document(d["file_hash"])

        texts = [c.page_content for c in chunks]
        dense = self._store.embeddings.embed_documents(texts)
        sparse = self._store.sparse.embed_documents(texts)
        points = [
            models.PointStruct(
                id=self._point_id(file_hash, "chunk", i),
                vector={"dense": dense[i], "bm25": sparse[i]},
                payload={
                    "tenant_id": self.tenant_id, "file_hash": file_hash, "source": source,
                    "text": chunk.page_content, "metadata": dict(chunk.metadata or {}),
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        for batch in _batches(points, _UPSERT_BATCH):
            self._client.upsert(self._store.chunks, batch, wait=True)

        page_points = [
            models.PointStruct(
                id=self._point_id(file_hash, "page", i), vector={},
                payload={"tenant_id": self.tenant_id, "kind": "page", "file_hash": file_hash,
                         "source": source, "page": i, "text": text},
            )
            for i, text in enumerate(pages or [])
        ]
        for batch in _batches(page_points, _UPSERT_BATCH):
            self._client.upsert(self._store.pages, batch, wait=True)

        # En dernier : le document n'apparaît qu'une fois entièrement écrit.
        self._client.upsert(self._store.pages, [models.PointStruct(
            id=self._point_id(file_hash, "document"), vector={},
            payload={"tenant_id": self.tenant_id, "kind": "document", "file_hash": file_hash,
                     "source": source, "chunk_count": len(chunks), "page_count": len(pages or []),
                     "created_at": time.time()},
        )], wait=True)
        logger.info(f"Qdrant [{self.tenant_id}] {source}: {len(chunks)} chunks, {len(pages or [])} pages")
        return len(chunks)

    def delete_document(self, file_hash: str) -> None:
        flt = self._filter(self._eq("file_hash", file_hash))
        for collection in (self._store.chunks, self._store.pages):
            self._client.delete(collection, points_selector=models.FilterSelector(filter=flt), wait=True)

    def delete_all(self) -> None:
        """Vide l'espace du tenant (et seulement le sien)."""
        for collection in (self._store.chunks, self._store.pages):
            self._client.delete(collection, points_selector=models.FilterSelector(filter=self._filter()), wait=True)

    # --- lecture --------------------------------------------------------------

    def pages(self) -> Dict[str, List[str]]:
        """Pages OCR par source, dans l'ordre : le format attendu par PageStore."""
        by_source: Dict[str, Dict[int, str]] = {}
        for r in self._scroll(self._store.pages, self._filter(self._eq("kind", "page"))):
            by_source.setdefault(r.payload["source"], {})[int(r.payload["page"])] = r.payload["text"]
        return {s: [p[i] for i in sorted(p)] for s, p in by_source.items()}

    def hybrid_search(self, query: str, bm25_k: int, vector_k: int, weights=(0.5, 0.5),
                      sources: Optional[List[str]] = None) -> List[Document]:
        """
        BM25 sparse + dense, fusion RRF côté Qdrant. `weights` = (bm25, dense), comme
        HYBRID_RETRIEVER_WEIGHTS ; k=60 comme l'EnsembleRetriever de LangChain qu'il remplace.
        `sources` restreint au(x) document(s) choisi(s) par le routeur.
        """
        extra = [models.FieldCondition(key="source", match=models.MatchAny(any=list(sources)))] if sources else []
        flt = self._filter(*extra)

        prefetch, fusion_weights = [], []
        sparse = self._store.sparse.embed_query(query)
        if sparse.indices:  # une requête faite de stopwords n'a pas de vecteur BM25
            prefetch.append(models.Prefetch(query=sparse, using="bm25", limit=bm25_k, filter=flt))
            fusion_weights.append(float(weights[0]))
        dense = self._store.embeddings.embed_query(query)
        prefetch.append(models.Prefetch(query=dense, using="dense", limit=vector_k, filter=flt))
        fusion_weights.append(float(weights[1]))

        result = self._client.query_points(
            self._store.chunks,
            prefetch=prefetch,
            query=models.RrfQuery(rrf=models.Rrf(k=60, weights=fusion_weights)),
            query_filter=flt,
            limit=bm25_k + vector_k,
            with_payload=True,
        )
        return [
            Document(page_content=p.payload["text"], metadata=dict(p.payload.get("metadata") or {}))
            for p in result.points
        ]


_store: Optional[QdrantStore] = None
_store_lock = threading.Lock()


def get_store() -> QdrantStore:
    """Instance partagée par le process (connexion et schéma vérifiés une seule fois)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = QdrantStore()
        return _store
