"""
Multi-Query Retrieval.
Génère plusieurs reformulations de la question pour améliorer le recall.
"""
import hashlib
from typing import List, Set
from langchain_core.documents import Document
from ..llm.models import small_llm
from ..config.settings import settings
from ..utils.logging import logger


MULTI_QUERY_PROMPT = """Tu es un expert en recherche documentaire. Génère {count} reformulations TRÈS DIFFÉRENTES de la question suivante.

Stratégies à utiliser (une par reformulation) :
1. Synonymes et paraphrases (reformuler avec d'autres mots)
2. Termes techniques anglais si la question est en français (ou inversement)
3. Focus sur un aspect spécifique de la question (plus précis)
4. Formulation inversée (décrire ce qu'on cherche plutôt que poser une question)

Question originale: {question}

Réponds UNIQUEMENT avec les {count} reformulations, une par ligne, sans numérotation ni explication."""


class MultiQueryRetriever:
    """
    Retriever qui génère plusieurs reformulations de la question,
    exécute chaque query, et fusionne les résultats.
    """

    def __init__(self, base_retriever, llm=None):
        """
        Args:
            base_retriever: Le retriever de base
            llm: LLM pour générer les reformulations (optionnel)
        """
        self.retriever = base_retriever
        self.llm = llm
        self.enabled = settings.MULTI_QUERY_ENABLED
        self.query_count = settings.MULTI_QUERY_COUNT

    def _get_llm(self):
        """Initialise le LLM si nécessaire."""
        if self.llm is None:
            # Un peu de variabilité pour les reformulations
            self.llm = small_llm(max_tokens=500, temperature=0.7)
        return self.llm

    def _generate_queries(self, question: str) -> List[str]:
        """
        Génère des reformulations de la question.

        Args:
            question: La question originale

        Returns:
            Liste de questions (originale + reformulations)
        """
        if not self.enabled or self.query_count <= 1:
            return [question]

        try:
            llm = self._get_llm()
            prompt = MULTI_QUERY_PROMPT.format(
                count=self.query_count,
                question=question
            )
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Parser les reformulations
            reformulations = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith(("-", "*", "#"))
            ]

            # Ajouter la question originale en premier
            queries = [question] + reformulations[:self.query_count]

            logger.debug(f"MultiQuery: {len(queries)} queries générées")
            return queries

        except Exception as e:
            logger.warning(f"MultiQuery: erreur génération, utilisation query originale: {e}")
            return [question]

    def _deduplicate_and_rank(self, all_docs: List[Document]) -> List[Document]:
        """
        Déduplique et trie par fréquence de retrieval.
        Un doc trouvé par plusieurs queries est plus probablement pertinent.
        """
        from collections import OrderedDict

        doc_freq: dict[str, int] = {}
        doc_map: OrderedDict[str, Document] = OrderedDict()

        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            doc_freq[content_hash] = doc_freq.get(content_hash, 0) + 1
            if content_hash not in doc_map:
                doc_map[content_hash] = doc

        # Trier par fréquence décroissante (docs trouvés par le plus de queries en premier)
        sorted_hashes = sorted(doc_map.keys(), key=lambda h: doc_freq[h], reverse=True)
        ranked_docs = []
        for h in sorted_hashes:
            doc = doc_map[h]
            doc.metadata["retrieval_freq"] = doc_freq[h]
            ranked_docs.append(doc)

        return ranked_docs

    def invoke(self, query: str) -> List[Document]:
        """
        Exécute la recherche multi-query.
        Les résultats sont triés par fréquence de retrieval (docs trouvés
        par le plus de queries en premier) pour optimiser le reranking.
        """
        if not self.enabled:
            return self.retriever.invoke(query)

        # Générer les reformulations
        queries = self._generate_queries(query)

        # Exécuter les queries en parallèle
        all_docs: List[Document] = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch_docs(q: str) -> List[Document]:
            try:
                return self.retriever.invoke(q)
            except Exception as e:
                logger.warning(f"MultiQuery: erreur pour query '{q[:50]}...': {e}")
                return []

        # Propager les ContextVar (périmètre du routage par document) aux threads :
        # ThreadPoolExecutor ne le fait pas par défaut, et les sous-requêtes
        # chercheraient alors dans TOUS les documents au lieu du périmètre choisi.
        import contextvars
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(contextvars.copy_context().run, fetch_docs, q): q
                for q in queries
            }
            for future in as_completed(futures):
                all_docs.extend(future.result())

        # Dédupliquer ET trier par fréquence
        ranked_docs = self._deduplicate_and_rank(all_docs)

        logger.info(
            f"MultiQuery: {len(queries)} queries -> "
            f"{len(all_docs)} docs -> {len(ranked_docs)} uniques "
            f"(top freq={ranked_docs[0].metadata.get('retrieval_freq', 0) if ranked_docs else 0})"
        )

        return ranked_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Alias pour compatibilité LangChain."""
        return self.invoke(query)
