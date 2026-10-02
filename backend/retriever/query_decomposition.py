"""
Query Decomposition Retriever.
Décompose les questions complexes en sous-questions pour améliorer le recall.
"""
from typing import List, Set
from langchain_core.documents import Document
from ..llm.models import large_llm
from ..config.settings import settings
from ..utils.logging import logger


DECOMPOSITION_PROMPT = """Tu es un assistant spécialisé dans la décomposition de questions complexes.

Question: {question}

Cette question est-elle complexe (plusieurs aspects, comparaisons, questions multiples) ?
Si OUI, décompose-la en 2-3 sous-questions simples et indépendantes.
Si NON, réponds "SIMPLE".

Format de réponse:
- Si complexe: une sous-question par ligne, sans numérotation
- Si simple: juste le mot "SIMPLE"

Réponse:"""


class QueryDecompositionRetriever:
    """
    Retriever qui décompose les questions complexes en sous-questions.

    Workflow:
    1. Analyse si la question est complexe
    2. Si oui, décompose en sous-questions
    3. Exécute chaque sous-question
    4. Fusionne et déduplique les résultats
    """

    def __init__(self, base_retriever, llm=None):
        """
        Args:
            base_retriever: Le retriever de base
            llm: LLM pour la décomposition
        """
        self.retriever = base_retriever
        self.llm = llm
        self.enabled = settings.QUERY_DECOMPOSITION_ENABLED

    def _get_llm(self):
        """Initialise le LLM si nécessaire."""
        if self.llm is None:
            self.llm = large_llm(max_tokens=300, temperature=0.3)
        return self.llm

    def _decompose_query(self, question: str) -> List[str]:
        """
        Décompose une question complexe en sous-questions.

        Args:
            question: La question originale

        Returns:
            Liste de questions (originale si simple, sous-questions si complexe)
        """
        if not self.enabled:
            return [question]

        try:
            llm = self._get_llm()
            prompt = DECOMPOSITION_PROMPT.format(question=question)
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Question simple
            if content.upper() == "SIMPLE" or len(content) < 20:
                logger.debug("QueryDecomposition: question simple, pas de décomposition")
                return [question]

            # Parser les sous-questions
            sub_questions = [
                line.strip()
                for line in content.split("\n")
                if line.strip()
                and not line.strip().startswith(("-", "*", "#"))
                and len(line.strip()) > 10
            ]

            if not sub_questions:
                return [question]

            # Ajouter la question originale pour ne pas perdre le contexte global
            all_questions = [question] + sub_questions[:3]  # Max 3 sous-questions

            logger.info(f"QueryDecomposition: question décomposée en {len(all_questions)} queries")
            return all_questions

        except Exception as e:
            logger.warning(f"QueryDecomposition: erreur, utilisation query originale: {e}")
            return [question]

    def _deduplicate_docs(self, all_docs: List[Document]) -> List[Document]:
        """Déduplique les documents par contenu."""
        seen_content: Set[str] = set()
        unique_docs: List[Document] = []

        for doc in all_docs:
            content_hash = hash(doc.page_content[:500])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    def invoke(self, query: str) -> List[Document]:
        """
        Exécute la recherche avec décomposition de query.

        Args:
            query: La question originale

        Returns:
            Liste de documents dédupliqués
        """
        if not self.enabled:
            return self.retriever.invoke(query)

        # Décomposer la question
        questions = self._decompose_query(query)

        if len(questions) == 1:
            # Question simple, pas besoin de fusion
            return self.retriever.invoke(query)

        # Exécuter les queries en parallèle
        all_docs: List[Document] = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch_docs(q: str) -> List[Document]:
            try:
                return self.retriever.invoke(q)
            except Exception as e:
                logger.warning(f"QueryDecomposition: erreur pour '{q[:50]}...': {e}")
                return []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fetch_docs, q): q for q in questions}
            for future in as_completed(futures):
                all_docs.extend(future.result())

        # Dédupliquer
        unique_docs = self._deduplicate_docs(all_docs)

        logger.info(
            f"QueryDecomposition: {len(questions)} queries -> "
            f"{len(all_docs)} docs -> {len(unique_docs)} uniques"
        )

        return unique_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Alias pour compatibilité LangChain."""
        return self.invoke(query)
