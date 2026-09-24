"""La chaîne de retrieval de production, montée par RetrieverBuilder, sans réseau.

BM25, Chroma (en mémoire), parent-child, multi-query, rerank et routage sont les vrais
composants ; seuls les modèles sont factices (embeddings déterministes, LLM et Cohere
scriptés). Ce qui est vérifié, c'est le câblage : un maillon retiré, réordonné ou qui perd
le périmètre du routeur en route fait échouer ces tests."""
import re

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from backend.config.settings import settings
from backend.document_processor.chunkers import ParentChildChunkingStrategy
from backend.document_processor.file_handler import DocumentProcessor
from backend.retriever import builder as B
from backend.retriever.parent_child_retriever import ParentChildRetriever
from conftest import FakeRetriever, make_doc

FILLER = " ".join(f"Paragraph {i} discusses general corporate matters and governance." for i in range(40))

CORPUS = {
    "/data/AMD_2022_10K.pdf": FILLER + " The quick ratio of AMD depends on cash, receivables and current "
                              "liabilities of 6,369 million. " + FILLER,
    "/data/BOEING_2022_10K.pdf": FILLER + " Boeing quick ratio and current liabilities are discussed in "
                                 "liquidity. Legal proceedings include the 737 MAX litigation. " + FILLER,
}


class ScriptedLLM:
    """Répond selon le prompt : reformulations pour multi-query, 'ALL' pour le routeur."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))

        class _R:
            content = "ALL"

        if "reformulations" in str(prompt):
            _R.content = "current liabilities\nquick ratio liquidity"
        return _R()


class FakeCohere:
    """Score = recouvrement lexical avec la requête, pour un ordre déterministe."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def rerank(self, model, query, documents, top_n):
        self.calls.append(len(documents))
        if self.fail:
            raise RuntimeError("cohere 503")
        q = set(re.findall(r"\w+", query.lower()))

        def score(text):
            return len(q & set(re.findall(r"\w+", text.lower()))) / (len(q) or 1)

        ranked = sorted(range(len(documents)), key=lambda i: score(documents[i]), reverse=True)[:top_n]

        class _Res:
            def __init__(self, i):
                self.index, self.relevance_score = i, score(documents[i])

        class _Resp:
            results = [_Res(i) for i in ranked]
            meta = None

        return _Resp()


def _chunks():
    strat = ParentChildChunkingStrategy(parent_chunk_size=600, child_chunk_size=150, child_overlap=30)
    out = []
    for source, text in CORPUS.items():
        out.extend(strat.split(text, {"source": source}))
    return out


@pytest.fixture
def chain(monkeypatch):
    monkeypatch.setattr(settings, "PARENT_CHILD_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_COUNT", 2)
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "DOCUMENT_ROUTING_ENABLED", True)
    for flag in ("HYDE_ENABLED", "QUERY_DECOMPOSITION_ENABLED", "CONTEXTUAL_COMPRESSION_ENABLED"):
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "CHROMA_COLLECTION_NAME", "test-chain")

    builder = B.RetrieverBuilder.__new__(B.RetrieverBuilder)
    builder.embeddings = DeterministicFakeEmbedding(size=64)
    builder.llm = builder.llm_text = ScriptedLLM()
    retriever = builder.build_hybrid_retriever(_chunks())
    cohere = FakeCohere()
    retriever.retriever._client = cohere  # DocumentRouterRetriever -> RerankRetriever
    return retriever, cohere, builder.llm_text


def test_chain_is_wired_in_the_expected_order(chain):
    retriever, _, _ = chain
    names = []
    node = retriever
    while node is not None:
        names.append(type(node).__name__)
        node = getattr(node, "retriever", None)
    assert names == [
        "DocumentRouterRetriever", "RerankRetriever", "MultiQueryRetriever",
        "ParentChildRetriever", "ScopedHybridRetriever",
    ]


def test_named_company_restricts_every_subquery_to_its_document(chain):
    """Le périmètre (ContextVar) doit traverser les threads de multi-query jusqu'à BM25 et Chroma :
    Boeing parle aussi de quick ratio, il ne doit jamais remonter pour une question sur AMD."""
    retriever, cohere, llm = chain
    docs = retriever.invoke("What is the quick ratio of AMD?")
    assert docs
    assert {d.metadata["source"] for d in docs} == {"/data/AMD_2022_10K.pdf"}
    assert any("reformulations" in p for p in llm.prompts), "multi-query n'a pas tourné"
    assert cohere.calls, "le rerank n'a pas tourné"


def test_results_are_reranked_parents_with_the_evidence_first(chain):
    retriever, _, _ = chain
    docs = retriever.invoke("What is the quick ratio of AMD?")
    assert "6,369" in docs[0].page_content
    scores = [d.metadata["rerank_score"] for d in docs]
    assert scores == sorted(scores, reverse=True)
    assert all("parent_content" not in d.metadata for d in docs)  # parents, pas des enfants
    assert len({d.page_content for d in docs}) == len(docs)


def test_unnamed_question_searches_all_documents(chain):
    retriever, _, _ = chain
    docs = retriever.invoke("Which company reports legal proceedings about the 737 MAX?")
    assert "737 MAX" in docs[0].page_content


def test_rerank_failure_keeps_the_candidates_instead_of_failing(chain):
    retriever, _, _ = chain
    retriever.retriever._client = FakeCohere(fail=True)
    docs = retriever.invoke("What is the quick ratio of AMD?")
    assert docs and all("rerank_score" not in d.metadata for d in docs)


def test_rerank_caps_candidates_sent_to_cohere(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    rr = B.RerankRetriever(FakeRetriever(default=[make_doc(f"d{i}") for i in range(100)]), None, None)
    rr._client = FakeCohere()
    rr.invoke("d1")
    assert rr._client.calls == [40]


def test_parent_child_orders_parents_by_matched_children():
    children = [
        make_doc("c1", source="A"), make_doc("c2", source="A"), make_doc("c3", source="A"), make_doc("orphan"),
    ]
    for c, pid in zip(children[:3], ["p1", "p2", "p2"]):
        c.metadata.update(parent_id=pid, parent_content=f"parent {pid}", is_child=True)
    out = ParentChildRetriever(FakeRetriever(default=children)).invoke("q")
    assert [d.page_content for d in out] == ["parent p2", "parent p1", "orphan"]
    assert out[0].metadata["matched_children_count"] == 2


def test_attach_pages_maps_chunks_to_their_ocr_page():
    pages = ["Cover page", "Item 7 revenue grew 44%", "Balance sheet current liabilities 6,369"]
    markdown = "\n\n".join(pages)
    spans, cursor = [], 0
    for i, p in enumerate(pages):
        spans.append((cursor, cursor + len(p), i))
        cursor += len(p) + 2
    chunks = [make_doc("Item 7 revenue"), make_doc("current liabilities 6,369"), make_doc("absent du document")]
    DocumentProcessor._attach_pages(chunks, markdown, spans)
    assert [c.metadata.get("page") for c in chunks] == [1, 2, None]
