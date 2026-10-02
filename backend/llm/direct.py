"""
Accès direct aux modèles, sans Amazon Bedrock : API Anthropic pour Claude, API Cohere pour
les embeddings. Même interface que bedrock.py ; le choix se fait dans models.py.

Les embeddings sont Cohere Embed v4 en 1024 dimensions, comme sur Bedrock : un index Qdrant
construit avec un fournisseur reste valable avec l'autre.
"""
from functools import lru_cache
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings

from ..config.settings import settings


def _api_key() -> str:
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY manquante : requise avec MODEL_PROVIDER=direct.")
    return settings.ANTHROPIC_API_KEY


def chat_model(model_id: str, max_tokens: int, temperature: float = 0.0) -> ChatAnthropic:
    """Modèle sans réflexion (Haiku 4.5) : classification, reformulation, routage."""
    return ChatAnthropic(
        model=model_id,
        api_key=_api_key(),
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def reasoning_llm() -> ChatAnthropic:
    """Modèle de raisonnement : réflexion adaptative et effort, sans `temperature`."""
    return ChatAnthropic(
        model=settings.REASONING_MODEL_ID,
        api_key=_api_key(),
        max_tokens=settings.REASONING_MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.REASONING_EFFORT},
        timeout=settings.REASONING_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def _cohere():
    import cohere

    if not settings.COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY manquante : requise pour les embeddings avec MODEL_PROVIDER=direct.")
    return cohere.ClientV2(api_key=settings.COHERE_API_KEY)


class CohereEmbeddings(Embeddings):
    """Cohere Embed v4 par l'API Cohere. Documents et requêtes ont chacun leur input_type."""

    BATCH = 96  # limite de textes par appel Cohere

    def __init__(self, model_id: str = None, dimensions: int = None):
        self.model_id = model_id or settings.EMBEDDING_MODEL_ID
        self.dimensions = dimensions or settings.EMBEDDING_DIMENSIONS

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.BATCH):
            response = _cohere().embed(
                model=self.model_id,
                texts=texts[i:i + self.BATCH],
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=self.dimensions,
                truncate="END",
            )
            out.extend(response.embeddings.float_)
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts), "search_document") if texts else []

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "search_query")[0]


def embeddings() -> Embeddings:
    return CohereEmbeddings()
