"""Propriétés (hypothesis) : des invariants qui doivent tenir pour N'IMPORTE QUEL texte,
pas seulement pour l'exemple choisi à la main. Tout tourne hors ligne."""
import re

from hypothesis import given, settings as hsettings, strategies as st

from backend.agents.corrective_retrieval import CorrectiveRetrieval
from backend.document_processor.chunkers import ParentChildChunkingStrategy, SemanticParentChildChunkingStrategy
from backend.retriever.multi_query import MultiQueryRetriever
from backend.retriever.page_store import PageStore
from conftest import FakeRetriever, make_doc

WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789$,.%()", min_size=1, max_size=12)
SENTENCE = st.lists(WORD, min_size=1, max_size=25).map(" ".join)
PARAGRAPH = st.lists(SENTENCE, min_size=1, max_size=6).map("\n".join)
DOCUMENT = st.lists(PARAGRAPH, min_size=1, max_size=25).map("\n\n".join)


def _words(text):
    return re.findall(r"\S+", text)


def _assert_parent_child_invariants(children, text, child_size):
    parents = {}
    for c in children:
        assert c.metadata["is_child"] is True
        assert c.page_content in c.metadata["parent_content"], "un enfant doit être extrait de son parent"
        assert len(c.page_content) <= child_size
        parents.setdefault(c.metadata["parent_id"], c.metadata["parent_content"])
    # Aucun mot du document ne disparaît du contexte renvoyé au modèle.
    covered = set(_words(" ".join(parents.values())))
    assert set(_words(text)) <= covered


@hsettings(max_examples=60, deadline=None)
@given(DOCUMENT)
def test_parent_child_chunking_loses_nothing(text):
    children = ParentChildChunkingStrategy(parent_chunk_size=300, child_chunk_size=80, child_overlap=10).split(text)
    _assert_parent_child_invariants(children, text, 80)


class _HashEmbeddings:
    """Embeddings déterministes à 8 dimensions : suffisant pour déclencher des ruptures."""

    def embed_documents(self, texts):
        return [[float((hash(t) >> (4 * i)) & 15) - 7.5 for i in range(8)] for t in texts]


@hsettings(max_examples=60, deadline=None)
@given(DOCUMENT)
def test_semantic_parent_child_chunking_loses_nothing(text):
    strat = SemanticParentChildChunkingStrategy(embeddings=_HashEmbeddings())
    strat.parent_max_size, strat.child_chunk_size = 300, 80
    strat._child_splitter._chunk_size, strat._child_splitter._chunk_overlap = 80, 10
    _assert_parent_child_invariants(strat.split(text), text, 80)


CONTENT = st.sampled_from([f"passage {i}" for i in range(12)])
RANKED_LISTS = st.lists(st.lists(CONTENT, max_size=10, unique=True), min_size=1, max_size=5)


@given(RANKED_LISTS, st.integers(min_value=1, max_value=15))
def test_rrf_merge_is_a_dedup_ranking(lists, top_n):
    merged = CorrectiveRetrieval._merge([[make_doc(t) for t in lst] for lst in lists], top_n)
    contents = [d.page_content for d in merged]
    assert len(contents) == len(set(contents))
    assert len(contents) == min(top_n, len({t for lst in lists for t in lst}))
    # Un passage classé premier par toutes les requêtes est premier après fusion.
    firsts = {lst[0] for lst in lists if lst}
    if len(firsts) == 1 and all(lists):
        assert contents[0] == firsts.pop()


@given(RANKED_LISTS)
def test_rrf_merge_does_not_depend_on_query_order(lists):
    fwd = CorrectiveRetrieval._merge([[make_doc(t) for t in lst] for lst in lists], 50)
    rev = CorrectiveRetrieval._merge([[make_doc(t) for t in lst] for lst in reversed(lists)], 50)
    rrf = {}
    for lst in lists:
        for rank, t in enumerate(lst, start=1):
            rrf[t] = round(rrf.get(t, 0.0) + 1 / (60 + rank), 12)
    # Mêmes passages, et même ordre aux ex-aequo près.
    assert [rrf[d.page_content] for d in fwd] == [rrf[d.page_content] for d in rev]
    assert {d.page_content for d in fwd} == {d.page_content for d in rev}


@given(st.lists(CONTENT, max_size=40))
def test_multi_query_dedup_ranks_by_frequency_and_keeps_everything(contents):
    mq = MultiQueryRetriever(FakeRetriever())
    ranked = mq._deduplicate_and_rank([make_doc(t) for t in contents])
    assert [d.page_content for d in ranked] == list(dict.fromkeys(
        sorted(dict.fromkeys(contents), key=lambda t: -contents.count(t))
    ))
    freqs = [d.metadata["retrieval_freq"] for d in ranked]
    assert freqs == sorted(freqs, reverse=True)


PAGE = st.lists(SENTENCE, max_size=5).map("\n".join)


@given(st.lists(PAGE, min_size=1, max_size=12), st.text(alphabet="abcdefghij", min_size=1, max_size=3))
def test_grep_is_exhaustive(pages, term):
    """README : « 0 résultat permet d'affirmer qu'un terme n'y figure pas ». Il faut que ce soit vrai."""
    store = PageStore({"DOC.pdf": pages})
    out = store.grep(term, "DOC", max_hits=10_000)
    expected = sum(1 for p in pages for line in p.splitlines() if term in line.lower())
    assert out["total"] == expected
    assert {h["page"] for h in out["hits"]} == {i for i, p in enumerate(pages) if term in p.lower()}
