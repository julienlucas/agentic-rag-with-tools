from langchain_core.embeddings import Embeddings
from ..llm.models import get_embeddings as _provider_embeddings
from ..utils.logging import logger


def get_embeddings() -> Embeddings:
    """Embeddings Cohere Embed v4, par Bedrock ou par l'API Cohere selon MODEL_PROVIDER."""
    from ..config.settings import settings
    logger.info(f"Embeddings Cohere Embed v4 ({settings.MODEL_PROVIDER})")
    return _provider_embeddings()
