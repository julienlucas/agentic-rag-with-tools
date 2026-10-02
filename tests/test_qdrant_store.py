"""
Isolation des espaces utilisateurs dans Qdrant, sans réseau (Qdrant en mémoire).

Le filtre tenant_id est la seule barrière entre deux utilisateurs : chaque opération
(recherche, pages, registre, suppression) est vérifiée contre un second tenant.
"""
import pytest

from backend.config.settings import settings
from backend.vectorstore import QuotaExceeded
from conftest import make_doc, make_store


@pytest.fixture
def store():
    return make_store()


def _index(tenant, file_hash, source, texts, pages=("page 1",)):
    return tenant.add_document(file_hash, source, [make_doc(t, source=source) for t in texts], list(pages))


def test_search_never_returns_another_tenants_chunks(store):
    alice, bob = store.for_tenant("alice"), store.for_tenant("bob")
    _index(alice, "ha", "alice.pdf", ["alice revenue was 10 million"])
    _index(bob, "hb", "bob.pdf", ["bob revenue was 99 million"])

    docs = alice.hybrid_search("revenue million", bm25_k=10, vector_k=10)
    assert docs and {d.metadata["source"] for d in docs} == {"alice.pdf"}
    # Même en ciblant explicitement le document de bob.
    assert alice.hybrid_search("revenue", bm25_k=10, vector_k=10, sources=["bob.pdf"]) == []


def test_registry_and_pages_are_per_tenant(store):
    alice, bob = store.for_tenant("alice"), store.for_tenant("bob")
    _index(alice, "ha", "alice.pdf", ["a"], pages=["p0", "p1"])
    assert alice.sources() == ["alice.pdf"]
    assert alice.pages() == {"alice.pdf": ["p0", "p1"]}
    assert bob.documents() == [] and bob.pages() == {} and bob.chunk_count() == 0
    assert not bob.has_document("ha")


def test_deletes_only_touch_the_own_space(store):
    alice, bob = store.for_tenant("alice"), store.for_tenant("bob")
    _index(alice, "same", "a.pdf", ["alice text"])
    _index(bob, "same", "b.pdf", ["bob text"])  # même fichier chez les deux
    bob.delete_document("same")
    bob.delete_all()
    assert alice.has_document("same") and alice.chunk_count() == 1
    alice.delete_all()
    assert alice.documents() == [] and alice.chunk_count() == 0


def test_reindexing_the_same_file_is_idempotent(store):
    t = store.for_tenant("t")
    _index(t, "h", "doc.pdf", ["one", "two"])
    _index(t, "h", "doc.pdf", ["one", "two"])
    assert t.chunk_count() == 2 and len(t.documents()) == 1


def test_new_version_of_a_file_replaces_the_old_one(store):
    t = store.for_tenant("t")
    _index(t, "v1", "doc.pdf", ["old one", "old two", "old three"], pages=["a", "b"])
    _index(t, "v2", "doc.pdf", ["new"], pages=["c"])
    assert [d["file_hash"] for d in t.documents()] == ["v2"]
    assert t.chunk_count() == 1 and t.pages() == {"doc.pdf": ["c"]}


def test_scope_restricts_search_to_the_routed_documents(store):
    t = store.for_tenant("t")
    _index(t, "a", "amd.pdf", ["quick ratio of amd"])
    _index(t, "b", "boeing.pdf", ["quick ratio of boeing"])
    docs = t.hybrid_search("quick ratio", bm25_k=10, vector_k=10, sources=["boeing.pdf"])
    assert {d.metadata["source"] for d in docs} == {"boeing.pdf"}


def test_quotas(store, monkeypatch):
    t = store.for_tenant("t")
    monkeypatch.setattr(settings, "TENANT_MAX_DOCUMENTS", 1)
    _index(t, "a", "a.pdf", ["x"])
    _index(t, "a2", "a.pdf", ["y"])  # remplacement : ne compte pas comme un document de plus
    with pytest.raises(QuotaExceeded):
        _index(t, "b", "b.pdf", ["z"])
    monkeypatch.setattr(settings, "TENANT_MAX_DOCUMENTS", 10)
    monkeypatch.setattr(settings, "TENANT_MAX_CHUNKS", 2)
    with pytest.raises(QuotaExceeded):
        _index(t, "c", "c.pdf", ["1", "2"])
    assert t.sources() == ["a.pdf"]  # rien d'écrit après un refus


@pytest.mark.parametrize("bad", ["", None, "a b", "x" * 129, "../etc", 42])
def test_invalid_tenant_ids_are_rejected(store, bad):
    with pytest.raises(ValueError):
        store.for_tenant(bad)
