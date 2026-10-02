from typing import List
from langchain_core.documents import Document
from ..llm.bedrock import large_llm
from ..config.settings import settings
from ..utils.logging import logger


HYDE_PROMPT = """Tu es un expert qui répond à des questions de manière détaillée et factuelle.

Question: {question}

Écris une réponse hypothétique détaillée à cette question comme si tu avais accès à tous les documents pertinents.
La réponse doit être factuelle, spécifique et contenir les termes techniques appropriés.
Ne dis pas "je ne sais pas" - imagine la réponse idéale basée sur tes connaissances.

Réponse hypothétique:"""


class HyDERetriever:
    """
    HyDE Retriever - Hypothetical Document Embeddings.

    Workflow:
    1. Génère une réponse hypothétique à la question
    2. Utilise cette réponse + la question originale pour la recherche
    3. Les embeddings de la réponse hypothétique matchent mieux les documents réels
    """

    def __init__(self, base_retriever, llm=None):
        """
        Args:
            base_retriever: Le retriever de base (hybrid, vector, etc.)
            llm: LLM pour générer la réponse hypothétique
        """
        self.retriever = base_retriever
        self.llm = llm
        self.enabled = settings.HYDE_ENABLED

    def _get_llm(self):
        """Initialise le LLM si nécessaire."""
        if self.llm is None:
            self.llm = large_llm(max_tokens=500, temperature=0.7)
        return self.llm

    def _generate_hypothetical_answer(self, question: str) -> str:
        """
        Génère une réponse hypothétique à la question.

        Args:
            question: La question originale

        Returns:
            Réponse hypothétique
        """
        try:
            llm = self._get_llm()
            prompt = HYDE_PROMPT.format(question=question)
            response = llm.invoke(prompt)
            hypothetical = response.content.strip()
            logger.debug(f"HyDE: réponse hypothétique générée ({len(hypothetical)} chars)")
            return hypothetical
        except Exception as e:
            logger.warning(f"HyDE: erreur génération, utilisation query originale: {e}")
            return ""

    def invoke(self, query: str) -> List[Document]:
        """
        Exécute la recherche HyDE.

        Args:
            query: La question originale

        Returns:
            Liste de documents
        """
        if not self.enabled:
            return self.retriever.invoke(query)

        # Générer la réponse hypothétique
        hypothetical = self._generate_hypothetical_answer(query)

        if hypothetical:
            # Combiner la question originale et la réponse hypothétique
            # La réponse hypothétique aide à matcher les termes techniques
            enhanced_query = f"{query}\n\n{hypothetical}"

            # Recherche avec la query enrichie
            docs = self.retriever.invoke(enhanced_query)
            logger.info(f"HyDE: recherche avec query enrichie -> {len(docs)} docs")
        else:
            # Fallback sur la query originale
            docs = self.retriever.invoke(query)

        return docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Alias pour compatibilité LangChain."""
        return self.invoke(query)
