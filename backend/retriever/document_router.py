"""
Routage par document.

Avant de chercher, on décide DANS QUEL(S) DOCUMENT(S) chercher. Quand plusieurs documents
longs sont indexés ensemble (ex. trois 10-K de 120 à 260 pages), la bonne page se noie parmi
des milliers de chunks concurrents ; restreindre la recherche au document que la question
désigne divise l'espace de recherche d'autant.

Deux composants :
- DocumentRouter        : question + liste des sources -> sous-ensemble de sources (ou None = toutes).
                          D'abord un matching déterministe sur le nom des fichiers (gratuit), puis
                          un LLM léger seulement si nécessaire.
- ScopedHybridRetriever : recherche hybride Qdrant (BM25 sparse + dense, RRF) dans l'espace du
                          tenant ; applique le périmètre courant (filtre `source` côté Qdrant).
- DocumentRouterRetriever : wrapper le plus externe ; route la question, pose le périmètre dans
                          une ContextVar, puis délègue à la chaîne habituelle.

Le périmètre circule par ContextVar plutôt que par argument : les wrappers intermédiaires
(ParentChild, MultiQuery, Rerank...) appellent `self.retriever.invoke(query)` sans rien savoir
du routage, et n'ont donc pas à être modifiés.
"""
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

from ..utils.logging import logger

# Périmètre de recherche courant : liste de sources, ou None pour "toutes".
RETRIEVAL_SCOPE: ContextVar[Optional[List[str]]] = ContextVar("retrieval_scope", default=None)

# Mots de fichiers trop génériques pour identifier un document.
_GENERIC_TOKENS = {
    "pdf", "docx", "txt", "md", "10k", "10q", "8k", "annual", "report", "rapport", "annuel",
    "document", "doc", "file", "fichier", "final", "copy", "copie", "version", "draft",
    "earnings", "q1", "q2", "q3", "q4", "fy", "the", "and", "inc", "corp", "co", "ltd",
}


def source_label(source: str) -> str:
    """Nom lisible d'une source (basename sans extension)."""
    base = os.path.basename(str(source))
    return os.path.splitext(base)[0]


def _identity_tokens(source: str) -> Set[str]:
    """
    Tokens qui identifient un document dans son nom de fichier.
    "AMERICANEXPRESS_2022_10K.pdf" -> {"americanexpress"} ; "Boeing 2022 10K" -> {"boeing"}.
    """
    label = source_label(source).lower()
    parts = re.split(r"[^a-z]+", label)
    return {p for p in parts if len(p) >= 3 and p not in _GENERIC_TOKENS}


@contextmanager
def retrieval_scope(sources: Optional[List[str]]):
    """Pose le périmètre de recherche pour la durée du bloc (thread-safe via ContextVar)."""
    token = RETRIEVAL_SCOPE.set(sources)
    try:
        yield
    finally:
        RETRIEVAL_SCOPE.reset(token)


ROUTER_PROMPT = """Tu dois choisir dans quel(s) document(s) chercher la réponse à une question.

Documents disponibles (un nom par ligne) :
{sources}

Question : {question}

Règles :
- Si la question désigne clairement un ou plusieurs documents (entreprise, sujet, période), réponds avec leurs noms EXACTS, un par ligne.
- Si la question ne permet pas de choisir, ou concerne tous les documents, réponds ALL.
- N'ajoute aucune explication."""


class DocumentRouter:
    """Décide du périmètre documentaire d'une question."""

    def __init__(self, sources: List[str], llm=None):
        self.sources = list(dict.fromkeys(sources))  # unique, ordre conservé
        self.llm = llm
        self._tokens: Dict[str, Set[str]] = {s: _identity_tokens(s) for s in self.sources}

    def _match_by_name(self, question: str) -> List[str]:
        """Matching déterministe : le nom du document apparaît-il dans la question ?"""
        q = re.sub(r"[^a-z]", "", question.lower())  # "American Express" -> "americanexpress"
        matched = []
        for source, tokens in self._tokens.items():
            if tokens and any(t in q for t in tokens):
                matched.append(source)
        return matched

    def _match_by_llm(self, question: str) -> Optional[List[str]]:
        if self.llm is None:
            return None
        labels = {source_label(s): s for s in self.sources}
        prompt = ROUTER_PROMPT.format(sources="\n".join(labels), question=question)
        try:
            content = self.llm.invoke(prompt).content.strip()
        except Exception as e:
            logger.warning(f"DocumentRouter: LLM indisponible ({e}), recherche sur tous les documents")
            return None
        if "ALL" in content.upper().split():
            return None
        chosen = []
        for line in content.splitlines():
            line = line.strip().strip("-•* ").strip()
            for label, source in labels.items():
                if line and (line == label or label.lower() in line.lower()):
                    chosen.append(source)
        return list(dict.fromkeys(chosen)) or None

    def route(self, question: str) -> Optional[List[str]]:
        """
        Retourne les sources à interroger, ou None pour toutes.
        Un seul document indexé -> pas de routage.
        """
        if len(self.sources) <= 1:
            return None
        matched = self._match_by_name(question)
        if matched:
            logger.info(f"DocumentRouter (nom): {[source_label(s) for s in matched]}")
            return matched
        chosen = self._match_by_llm(question)
        if chosen:
            logger.info(f"DocumentRouter (LLM): {[source_label(s) for s in chosen]}")
        else:
            logger.info("DocumentRouter: aucun document ciblé, recherche sur tous")
        return chosen


class ScopedHybridRetriever:
    """
    BM25 sparse + vecteurs denses, fusion RRF côté Qdrant, restreints à l'espace du tenant
    et au périmètre courant (filtre `source` posé par DocumentRouterRetriever).
    """

    def __init__(self, tenant_store, weights, bm25_k: int, vector_k: int):
        self.tenant_store = tenant_store
        self.weights = list(weights)
        self.bm25_k = bm25_k
        self.vector_k = vector_k

    def invoke(self, query: str) -> List[Document]:
        return self.tenant_store.hybrid_search(
            query, bm25_k=self.bm25_k, vector_k=self.vector_k,
            weights=self.weights, sources=RETRIEVAL_SCOPE.get(),
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)


class DocumentRouterRetriever:
    """Wrapper externe : route la question puis délègue à la chaîne dans ce périmètre."""

    def __init__(self, base_retriever, router: DocumentRouter):
        self.retriever = base_retriever
        self.router = router

    def route(self, question: str) -> Optional[List[str]]:
        return self.router.route(question)

    def invoke_with_scope(self, query: str, sources: Optional[List[str]]) -> List[Document]:
        """Cherche dans un périmètre imposé (utilisé par la recherche corrective)."""
        with retrieval_scope(sources):
            return self.retriever.invoke(query)

    def invoke(self, query: str) -> List[Document]:
        return self.invoke_with_scope(query, self.route(query))

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)
