import threading
from ..config.settings import settings
from ..llm.bedrock import small_llm
from ..utils.logging import logger
from .embeddings import get_embeddings
from .parent_child_retriever import ParentChildRetriever
from .multi_query import MultiQueryRetriever
from .hyde import HyDERetriever
from .query_decomposition import QueryDecompositionRetriever
from .contextual_compression import ContextualCompressionRetriever
from .sentence_window_retriever import SentenceWindowRetriever
from .document_router import DocumentRouter, DocumentRouterRetriever, ScopedHybridRetriever


class RetrieverBuilder:
    def __init__(self):
        """Initialiser le constructeur de récupérateur avec les embeddings."""
        self.embeddings = get_embeddings()
        self.llm = small_llm(max_tokens=10)
        # LLM "texte" pour les composants qui doivent produire plusieurs lignes
        # (reformulations multi-query, noms de documents du routeur). Le self.llm
        # ci-dessus est bridé à 10 tokens : passé à MultiQuery, il tronquait les
        # reformulations à quelques mots.
        self.llm_text = small_llm(max_tokens=200)

    def build_hybrid_retriever(self, tenant_store):
        """
        Construire la chaîne de retrieval sur l'espace Qdrant d'un tenant.

        Les documents sont déjà indexés (TenantStore.add_document) : la recherche hybride
        BM25 sparse + dense tourne côté Qdrant, filtrée sur le tenant.
        """
        weights = settings.HYBRID_RETRIEVER_WEIGHTS
        if len(weights) != 2:
            logger.warning(f"Poids incorrects: {weights}, utilisation des poids par défaut")
            weights = (0.5, 0.5)

        # Même fusion RRF que l'ancien EnsembleRetriever, capable de se restreindre au
        # périmètre posé par DocumentRouterRetriever (filtre `source` côté Qdrant).
        retriever = ScopedHybridRetriever(
            tenant_store, weights, bm25_k=settings.BM25_K, vector_k=settings.VECTOR_SEARCH_K,
        )
        logger.info(f"Récupérateur hybride Qdrant créé (tenant {tenant_store.tenant_id}).")

        # Chaîner les composants optimisés pour le recall :
        # Hybrid → SentenceWindow → ParentChild → HyDE → MultiQuery → QueryDecomp → Rerank → Compression

        # Sentence Window (étend les phrases au contexte de fenêtre)
        if settings.CHUNKING_STRATEGY.lower() == "sentence_window":
            retriever = SentenceWindowRetriever(retriever)
            logger.info("Sentence Window retriever activé.")

        # Parent-Child (retourne parents des children matchés)
        if settings.PARENT_CHILD_ENABLED:
            retriever = ParentChildRetriever(retriever)
            logger.info("Parent-Child retriever activé.")

        # HyDE (génère une réponse hypothétique pour améliorer les embeddings)
        if settings.HYDE_ENABLED:
            retriever = HyDERetriever(retriever, self.llm)
            logger.info("HyDE retriever activé.")

        # Multi-Query (génère variations de la question)
        if settings.MULTI_QUERY_ENABLED:
            retriever = MultiQueryRetriever(retriever, self.llm_text)
            logger.info("Multi-Query retriever activé.")

        # Query Decomposition (décompose les questions complexes)
        if settings.QUERY_DECOMPOSITION_ENABLED:
            retriever = QueryDecompositionRetriever(retriever, self.llm)
            logger.info("Query Decomposition retriever activé.")

        # Reranker (améliore la précision du ranking)
        if settings.RERANK_ENABLED:
            retriever = RerankRetriever(retriever, self.embeddings, self.llm)
            logger.info(f"Reranker activé: {settings.RERANK_MODEL}")

        # Contextual Compression (filtre le contenu non pertinent)
        if settings.CONTEXTUAL_COMPRESSION_ENABLED:
            retriever = ContextualCompressionRetriever(retriever, self.llm)
            logger.info("Contextual Compression activé.")

        # Routage par document (le plus externe : décide du périmètre avant tout)
        if settings.DOCUMENT_ROUTING_ENABLED:
            retriever = self._wrap_with_router(retriever, tenant_store.sources())

        return retriever

    def _wrap_with_router(self, retriever, sources):
        """Ajoute le routage par document si plusieurs sources sont indexées."""
        sources = list(dict.fromkeys(str(s) for s in sources if s))
        if len(sources) <= 1:
            logger.info("Routage par document: une seule source, désactivé.")
            return retriever
        router = DocumentRouter(sources, llm=self.llm_text)
        logger.info(f"Routage par document activé sur {len(sources)} sources.")
        return DocumentRouterRetriever(retriever, router)


# Unités de recherche facturées par Cohere (une requête, jusqu'à 100 documents), cumulées
# sur le process : l'évaluation FinanceBench les lit pour chiffrer le coût d'un run.
_RERANK_USAGE_LOCK = threading.Lock()
RERANK_USAGE = {"calls": 0, "search_units": 0.0}


class RerankRetriever:
    """
    Reranker utilisant l'API Cohere Rerank.
    Top-tier performance, excellent multilingue.
    """

    def __init__(self, retriever, embeddings, llm):
        self.retriever = retriever
        self.embeddings = embeddings
        self.llm = llm
        self._client = None
        logger.info(f"Cohere Reranker initialisé: {settings.RERANK_MODEL}")

    @property
    def client(self):
        """Lazy loading du client Cohere."""
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=settings.COHERE_API_KEY)
        return self._client

    def invoke(self, query: str):
        """Reranke les documents via l'API Cohere Rerank."""
        return self.rerank(query, self.retriever.invoke(query))

    def rerank(self, query: str, docs, top_n: int = None):
        """
        Reclasse `docs` selon `query`. Exposé séparément de invoke() pour que la recherche
        corrective puisse reclasser un ensemble fusionné contre la question d'ORIGINE :
        classés selon les requêtes réécrites, les passages ajoutés éjectaient des preuves
        que le retrieval initial avait bien trouvées.
        """
        if not docs or not settings.RERANK_ENABLED:
            return docs

        # Cap le nombre de candidats envoyés à Cohere pour limiter la latence.
        # Multi-query peut produire 100-200 docs uniques, ce qui ralentit
        # inutilement le rerank sans gain de qualité notable au-delà de ~40.
        MAX_RERANK_CANDIDATES = 40
        if len(docs) > MAX_RERANK_CANDIDATES:
            docs = docs[:MAX_RERANK_CANDIDATES]
        num_to_rerank = len(docs)

        # Compté avant l'appel : une requête tentée est une unité de recherche, qu'elle réussisse
        # ou non ; si Cohere renvoie le détail facturé, on corrige après.
        with _RERANK_USAGE_LOCK:
            RERANK_USAGE["calls"] += 1
            RERANK_USAGE["search_units"] += 1.0

        try:
            # Cohere rerank a une limite de ~10000 docs, largement suffisant
            response = self.client.rerank(
                model=settings.RERANK_MODEL,
                query=query,
                documents=[d.page_content for d in docs],
                top_n=min(top_n or settings.RERANK_TOP_K, num_to_rerank),
            )

            billed = getattr(getattr(response, "meta", None), "billed_units", None)
            units = getattr(billed, "search_units", None)
            if units is not None:
                with _RERANK_USAGE_LOCK:
                    RERANK_USAGE["search_units"] += float(units) - 1.0

            # Reconstruire la liste ordonnée par score de reranking
            reranked_docs = []
            for result in response.results:
                doc = docs[result.index]
                doc.metadata["rerank_score"] = result.relevance_score
                reranked_docs.append(doc)

            # Log des scores pour debug
            if response.results:
                top_score = response.results[0].relevance_score
                avg_score = sum(r.relevance_score for r in response.results) / len(response.results)
                logger.info(
                    f"Cohere Rerank: {num_to_rerank} docs -> top {len(reranked_docs)}, "
                    f"top_score={top_score:.3f}, avg={avg_score:.3f}"
                )

            return reranked_docs

        except Exception as e:
            logger.warning(f"Erreur Cohere Rerank: {e}, retour des docs non rerankés")
            return docs

    def get_relevant_documents(self, query: str):
        """Alias pour compatibilité LangChain."""
        return self.invoke(query)