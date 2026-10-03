"""
Accès aux modèles via Amazon Bedrock (MODEL_PROVIDER=bedrock). Même interface que direct.py ;
le choix se fait dans models.py.

- Chat (Claude) : API Converse via ChatBedrockConverse. Les identifiants sont des profils
  d'inférence régionaux (« eu.anthropic... ») : en eu-west-3, Haiku 4.5 et Sonnet refusent
  l'appel direct à l'ID du modèle (« on-demand throughput isn't supported »).
- Embeddings : Cohere Embed v4 (InvokeModel), même profil « eu. ».
Le reranker (Cohere Rerank 4 Pro) reste chez Cohere, par son API : voir retriever/builder.py.

Identifiants AWS : ceux du `.env` du projet s'il en contient, en PRIORITÉ sur l'environnement
du shell. Un terminal qui a exporté un AWS_SESSION_TOKEN expiré (profil SSO d'un autre compte)
faisait échouer chaque appel (« security token included in the request is invalid ») : ces
variables passent avant le `.env` pour load_dotenv et pydantic. Sans clés dans le `.env`
(déploiement avec rôle IAM, ~/.aws/credentials), chaîne standard de boto3.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import boto3
from dotenv import dotenv_values
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.embeddings import Embeddings

from ..config.settings import settings


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


@lru_cache(maxsize=1)
def _session() -> boto3.Session:
    env = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        # Session explicite : ignore AWS_PROFILE / AWS_SESSION_TOKEN / clés du shell.
        return boto3.Session(
            aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=env.get("AWS_SESSION_TOKEN") or None,
        )
    return boto3.Session()


def _region() -> str:
    env = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    return env.get("AWS_REGION") or settings.AWS_REGION


@lru_cache(maxsize=None)
def _client(service: str, region: str, read_timeout: int):
    # Clients boto3 thread-safe : partagés (multi-query appelle les modèles depuis des threads).
    # Les reprises sur throttling sont faites ici, par boto3 (mode adaptive).
    return _session().client(
        service,
        region_name=region,
        config=Config(
            read_timeout=read_timeout,
            connect_timeout=10,
            retries={"max_attempts": settings.LLM_MAX_RETRIES + 1, "mode": "adaptive"},
        ),
    )


def runtime_client(read_timeout: int = None):
    return _client("bedrock-runtime", _region(), read_timeout or settings.LLM_TIMEOUT)


def chat_model(model_id: str, max_tokens: int, temperature: float = 0.0,
               region: Optional[str] = None) -> ChatBedrockConverse:
    """
    Modèle sans réflexion (Haiku 4.5) : classification, reformulation, routage.
    `region` : pour un modèle absent de la région du projet (Ministral 3 de l'évaluation).
    """
    region = region or _region()
    return ChatBedrockConverse(
        model=model_id,
        client=_client("bedrock-runtime", region, settings.LLM_TIMEOUT),
        region_name=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def reasoning_llm() -> ChatBedrockConverse:
    """
    Modèle de raisonnement (génération avec outils + agent de recherche).
    Pas de `temperature` (les modèles récents la refusent). Réflexion adaptative et effort passent
    tels quels à l'API Anthropic par additionalModelRequestFields ; le contenu des réponses
    est alors une liste de blocs (reasoning_content, text, tool_use) : lire le texte avec
    search_agent.message_text.
    """
    return ChatBedrockConverse(
        model=settings.REASONING_MODEL_ID,
        client=runtime_client(settings.REASONING_TIMEOUT),
        region_name=_region(),
        max_tokens=settings.REASONING_MAX_TOKENS,
        additional_model_request_fields={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": settings.REASONING_EFFORT},
        },
    )


class BedrockCohereEmbeddings(Embeddings):
    """Cohere Embed v4 sur Bedrock. Documents et requêtes ont chacun leur input_type."""

    BATCH = 96  # limite de textes par appel Cohere

    def __init__(self, model_id: str = None, dimensions: int = None):
        self.model_id = model_id or settings.EMBEDDING_MODEL_ID
        self.dimensions = dimensions or settings.EMBEDDING_DIMENSIONS

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.BATCH):
            body = {
                "texts": texts[i:i + self.BATCH],
                "input_type": input_type,
                "embedding_types": ["float"],
                "output_dimension": self.dimensions,
                "truncate": "RIGHT",
            }
            response = runtime_client().invoke_model(modelId=self.model_id, body=json.dumps(body))
            out.extend(json.loads(response["body"].read())["embeddings"]["float"])
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts), "search_document") if texts else []

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "search_query")[0]


def embeddings() -> Embeddings:
    return BedrockCohereEmbeddings()
