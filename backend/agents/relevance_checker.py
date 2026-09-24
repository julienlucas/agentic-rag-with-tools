import re
from typing import Optional

from langchain_mistralai import ChatMistralAI
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class RelevanceChecker:
    def __init__(self):
        self.model = ChatMistralAI(
            model=settings.MODEL_SMALL_ID,
            api_key=settings.MISTRALAI_API_KEY,
            temperature=0,
            max_tokens=10,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    def check(self, question: str, documents, k=3) -> str:
        """
        1. Utiliser les documents déjà récupérés (pas de nouvel appel retriever).
        2. Les combiner en une seule chaîne de texte.
        3. Passer ce texte + question au LLM pour classification.

        Retourne: "CAN_ANSWER", "PARTIAL", ou "NO_MATCH".
        """

        logger.debug(f"RelevanceChecker.check appelé avec question='{question}' et k={k}")

        top_docs = documents
        if not top_docs:
            logger.debug("Aucun document fourni. Classification comme NO_MATCH.")
            return "NO_MATCH"

        # Combiner les k premiers chunks de texte en une seule chaîne
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        # Créer un prompt pour le LLM afin de classifier la pertinence
        prompt = f"""
        Vous êtes un vérificateur de pertinence IA entre la question d'un utilisateur et le contenu de document fourni.

        **Instructions:**
        - Classifiez dans quelle mesure le contenu du document répond à la question de l'utilisateur.
        - Répondez avec un seul des labels suivants: CAN_ANSWER, PARTIAL, NO_MATCH.
        - N'incluez aucun texte ou explication supplémentaire.

        **Labels:**
        1) "CAN_ANSWER": Les passages contiennent suffisamment d'informations explicites pour répondre complètement à la question.
        2) "PARTIAL": Les passages mentionnent ou discutent le sujet de la question mais ne fournissent pas tous les détails nécessaires pour une réponse complète.
        3) "NO_MATCH": Les passages ne discutent ni ne mentionnent le sujet de la question du tout.

        **Important:** Si les passages mentionnent ou font référence au sujet ou à la période de la question de quelque manière que ce soit, même si incomplète, répondez avec "PARTIAL" au lieu de "NO_MATCH".

        **Question:** {question}
        **Passages:** {document_content}

        **Répondez UNIQUEMENT avec un des labels suivants: CAN_ANSWER, PARTIAL, NO_MATCH**
        """

        # Appeler le LLM
        try:
            response = self.model.invoke(prompt)
        except Exception as e:
            from ..utils.resilience import is_rate_limit
            if is_rate_limit(e):
                raise  # à rejouer par l'appelant, pas un vrai NO_MATCH
            logger.error(f"Erreur lors de l'inférence du modèle: {e}")
            return "NO_MATCH"

        # Extraire le contenu de la réponse
        try:
            llm_response = response.content.strip()
            logger.debug(f"Réponse du LLM: {llm_response}")
        except (IndexError, KeyError) as e:
            logger.error(f"Structure de réponse inattendue: {e}")
            return "NO_MATCH"

        classification = parse_relevance_label(llm_response)
        if classification is None:
            logger.warning(f"Label de pertinence illisible ({llm_response[:80]!r}). Forçage de 'NO_MATCH'.")
            return "NO_MATCH"
        logger.debug(f"Classification reconnue comme '{classification}'.")
        return classification


_LABEL_RE = re.compile(r"(?<![A-Z_])(CAN[ _-]ANSWER|PARTIAL|NO[ _-]MATCH)(?![A-Z_])")


def parse_relevance_label(text: str) -> Optional[str]:
    """
    Extrait le label de la réponse du modèle, ou None s'il n'y en a pas.

    Une égalité stricte classait en NO_MATCH, sans bruit, des réponses pourtant claires :
    « CAN_ANSWER. », « **PARTIAL** », « Label: CAN_ANSWER ». Sans outils, NO_MATCH mène
    au refus. On prend le premier label cité, quelle que soit sa mise en forme.
    """
    m = _LABEL_RE.search((text or "").upper())
    if not m:
        return None
    return re.sub(r"[ -]", "_", m.group(1))
