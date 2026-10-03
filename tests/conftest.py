"""
Environnement minimal pour importer le backend sans clés réelles ni appel réseau.
Doit être importé avant backend.config.settings (pytest charge conftest en premier).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MISTRALAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
os.environ.setdefault("QDRANT_URL", "")  # tests : jamais le cluster réel
os.environ.setdefault("LANGSMITH_API_KEY", "")
os.environ.setdefault("FEEDBACK_AUTO_COLLECT", "false")  # pas de thread vers LangSmith en test

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


def make_doc(text: str, source: str = "DOC.pdf", page: int | None = None, rerank: float | None = None) -> Document:
    meta = {"source": source, "doc_name": source}
    if page is not None:
        meta["page"] = page
    if rerank is not None:
        meta["rerank_score"] = rerank
    return Document(page_content=text, metadata=meta)


class FakeLLM:
    """Remplace ChatMistralAI : renvoie des contenus prédéfinis, ou lève une exception."""

    def __init__(self, contents=None, error: Exception | None = None):
        self.contents = list(contents or [])
        self.error = error
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        content = self.contents.pop(0) if self.contents else ""

        class _Resp:
            pass

        r = _Resp()
        r.content = content
        return r


class FakeRetriever:
    """Retriever déterministe : une liste de docs par requête (ou par défaut)."""

    def __init__(self, by_query=None, default=None):
        self.by_query = by_query or {}
        self.default = default or []
        self.calls = []

    def invoke(self, query: str):
        self.calls.append(query)
        return list(self.by_query.get(query, self.default))


class FakeSparse:
    """Remplace fastembed BM25 : fréquences de termes, hashés de façon stable (sans réseau)."""

    @staticmethod
    def _vec(text: str, query: bool = False):
        import re
        import zlib
        from qdrant_client import models

        counts = {}
        for tok in re.findall(r"\w+", text.lower()):
            idx = zlib.crc32(tok.encode())
            counts[idx] = 1.0 if query else counts.get(idx, 0.0) + 1.0
        return models.SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text, query=True)


def make_store():
    """QdrantStore en mémoire, embeddings et BM25 factices."""
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from qdrant_client import QdrantClient

    from backend.config.settings import settings
    from backend.vectorstore import QdrantStore

    return QdrantStore(
        client=QdrantClient(":memory:"),
        embeddings=DeterministicFakeEmbedding(size=settings.EMBEDDING_DIMENSIONS),
        sparse=FakeSparse(),
    )


class FakeCohere:
    """Score = recouvrement lexical avec la requête, pour un ordre déterministe."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def rerank(self, model, query, documents, top_n):
        import re

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


@pytest.fixture
def docs():
    return make_doc, FakeLLM, FakeRetriever
