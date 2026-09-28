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
os.environ.setdefault("LANGSMITH_API_KEY", "")

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


@pytest.fixture
def docs():
    return make_doc, FakeLLM, FakeRetriever
