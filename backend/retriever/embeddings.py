from langchain_core.embeddings import Embeddings
from ..llm.bedrock import BedrockCohereEmbeddings
from ..utils.logging import logger


def get_embeddings() -> Embeddings:
    """Embeddings Cohere Embed v4 via Amazon Bedrock."""
    logger.info("Utilisation des embeddings Cohere Embed v4 (Bedrock)")
    return BedrockCohereEmbeddings()
