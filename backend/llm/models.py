"""
Point d'entrée unique vers les modèles : choisit Bedrock ou les API directes selon
MODEL_PROVIDER. Le reste du code n'importe que ce module.
"""
from langchain_core.embeddings import Embeddings

from ..config.settings import settings


def _provider():
    if settings.MODEL_PROVIDER == "direct":
        from . import direct
        return direct
    from . import bedrock
    return bedrock


def small_llm(max_tokens: int, temperature: float = 0.0):
    return _provider().chat_model(settings.MODEL_SMALL_ID, max_tokens, temperature)


def large_llm(max_tokens: int, temperature: float = 0.0):
    return _provider().chat_model(settings.MODEL_ID, max_tokens, temperature)


def reasoning_llm():
    return _provider().reasoning_llm()


def get_embeddings() -> Embeddings:
    return _provider().embeddings()
