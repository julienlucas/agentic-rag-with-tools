from langchain_anthropic import ChatAnthropic
from ..config.settings import settings


def reasoning_llm() -> ChatAnthropic:
    """Modèle de raisonnement (Claude Sonnet 5), partagé par la génération et l'agent de recherche."""
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY manquante : l'ajouter au .env (modèle de raisonnement Claude).")
    return ChatAnthropic(
        model=settings.REASONING_MODEL_ID,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=settings.REASONING_MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.REASONING_EFFORT},
        timeout=settings.REASONING_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
    )
