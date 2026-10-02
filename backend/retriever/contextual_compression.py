"""
Contextual Compression Retriever.
Extrait et compresse les passages les plus pertinents des documents récupérés.
Améliore la précision en filtrant le contenu non pertinent.
"""
from typing import List
from langchain_core.documents import Document
from ..llm.models import large_llm
from ..config.settings import settings
from ..utils.logging import logger


EXTRACTION_PROMPT = """Tu es un assistant qui extrait les passages pertinents d'un document.

Question: {question}

Document:
{document}

Extrais UNIQUEMENT les phrases ou paragraphes qui répondent directement à la question.
Si aucun passage n'est pertinent, réponds "AUCUN".
Ne reformule pas, copie les passages tels quels.

Passages pertinents:"""


class ContextualCompressionRetriever:
    """
    Retriever qui compresse les documents en extrayant uniquement les passages pertinents.

    Workflow:
    1. Récupère les documents via le retriever de base
    2. Pour chaque document, extrait les passages pertinents
    3. Filtre les documents sans passages pertinents
    """

    def __init__(self, base_retriever, llm=None):
        """
        Args:
            base_retriever: Le retriever de base
            llm: LLM pour l'extraction (optionnel)
        """
        self.retriever = base_retriever
        self.llm = llm
        self.enabled = settings.CONTEXTUAL_COMPRESSION_ENABLED
        self.min_relevance_length = 50  # Longueur min pour considérer un passage pertinent

    def _get_llm(self):
        """Initialise le LLM si nécessaire."""
        if self.llm is None:
            self.llm = large_llm(max_tokens=1000)
        return self.llm

    def _extract_relevant_content(self, question: str, document: Document) -> Document | None:
        """
        Extrait le contenu pertinent d'un document.

        Args:
            question: La question
            document: Le document à compresser

        Returns:
            Document compressé ou None si non pertinent
        """
        try:
            llm = self._get_llm()
            prompt = EXTRACTION_PROMPT.format(
                question=question,
                document=document.page_content[:2000]  # Limiter la taille
            )
            response = llm.invoke(prompt)
            extracted = response.content.strip()

            # Vérifier si le contenu est pertinent
            if (
                extracted.upper() == "AUCUN"
                or len(extracted) < self.min_relevance_length
                or "aucun passage" in extracted.lower()
            ):
                return None

            # Créer un nouveau document avec le contenu compressé
            compressed_doc = Document(
                page_content=extracted,
                metadata={
                    **document.metadata,
                    "compressed": True,
                    "original_length": len(document.page_content),
                    "compressed_length": len(extracted)
                }
            )
            return compressed_doc

        except Exception as e:
            logger.warning(f"Contextual Compression: erreur extraction: {e}")
            return document  # Retourner le document original en cas d'erreur

    def invoke(self, query: str) -> List[Document]:
        """
        Exécute la recherche avec compression contextuelle.

        Args:
            query: La question

        Returns:
            Liste de documents compressés
        """
        # Récupérer les documents de base
        docs = self.retriever.invoke(query)

        if not self.enabled or not docs:
            return docs

        # Compresser les top documents (limiter pour la latence)
        top_k_compress = min(settings.CONTEXTUAL_COMPRESSION_TOP_K, len(docs))
        docs_to_compress = docs[:top_k_compress]
        remaining_docs = docs[top_k_compress:]

        # Compression en parallèle pour la performance
        from concurrent.futures import ThreadPoolExecutor, as_completed

        compressed_docs: List[Document] = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._extract_relevant_content, query, doc): i
                for i, doc in enumerate(docs_to_compress)
            }

            results = [None] * len(docs_to_compress)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

            # Garder l'ordre et filtrer les None
            compressed_docs = [doc for doc in results if doc is not None]

        # Stats
        original_count = len(docs_to_compress)
        compressed_count = len(compressed_docs)
        filtered_count = original_count - compressed_count

        logger.info(
            f"Contextual Compression: {original_count} docs -> "
            f"{compressed_count} pertinents ({filtered_count} filtrés)"
        )

        return compressed_docs + remaining_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Alias pour compatibilité LangChain."""
        return self.invoke(query)
