"""
Le schéma Qdrant doit rester celui de la multi-tenancy : métrique explicite, HNSW par tenant.

Le graphe global est désactivé (m=0) et chaque tenant a le sien (payload_m) via l'index
`tenant_id` déclaré is_tenant. Un réglage retiré ne casse rien de visible (la recherche
marche toujours) : il dégrade l'isolation physique ou la mémoire. Ces tests le rendent visible.
"""
from qdrant_client import QdrantClient, models

from backend.config.settings import settings
from backend.vectorstore import QdrantStore
from conftest import make_store


def test_vector_space_is_explicit_and_valid():
    assert settings.VECTOR_SPACE in {"cosine", "l2", "ip"}
    assert settings.VECTOR_SPACE == "cosine"


class SpyClient(QdrantClient):
    """Qdrant local ignore HNSW et index de payload : on vérifie ce qui lui est DEMANDÉ."""

    def __init__(self):
        super().__init__(":memory:")
        self.collections, self.indexes = {}, []

    def create_collection(self, collection_name, **kwargs):
        self.collections[collection_name] = kwargs
        return super().create_collection(collection_name, **kwargs)

    def create_payload_index(self, collection_name, field_name, field_schema=None, **kwargs):
        self.indexes.append((collection_name, field_name, field_schema))
        return super().create_payload_index(collection_name, field_name, field_schema=field_schema, **kwargs)


def test_chunks_collection_schema():
    client = SpyClient()
    store = QdrantStore(client=client, embeddings=object(), sparse=object())
    spec = client.collections[store.chunks]
    dense = spec["vectors_config"]["dense"]
    assert dense.distance == models.Distance.COSINE
    assert dense.size == settings.EMBEDDING_DIMENSIONS
    assert spec["sparse_vectors_config"]["bm25"].modifier == models.Modifier.IDF
    assert spec["hnsw_config"].m == 0
    assert spec["hnsw_config"].payload_m == settings.QDRANT_HNSW_PAYLOAD_M


def test_tenant_index_is_declared_first_and_is_tenant():
    client = SpyClient()
    store = QdrantStore(client=client, embeddings=object(), sparse=object())
    for collection in (store.chunks, store.pages):
        first = next(i for i in client.indexes if i[0] == collection)
        assert first[1] == "tenant_id" and first[2].is_tenant is True


def test_existing_collection_with_other_dimension_is_refused(monkeypatch):
    store = make_store()
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSIONS", 256)
    try:
        type(store)(client=store._client, embeddings=store._embeddings, sparse=store.sparse)
    except RuntimeError as e:
        assert "dimension" in str(e)
    else:
        raise AssertionError("une collection de dimension différente doit être refusée")
