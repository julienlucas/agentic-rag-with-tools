"""MODEL_PROVIDER choisit Bedrock ou les API directes, sans appel réseau (construction seule)."""
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse

from backend.config.settings import PROVIDER_DEFAULTS, Settings, settings
from backend.llm import models


@pytest.fixture
def provider(monkeypatch):
    def use(name):
        monkeypatch.setattr(settings, "MODEL_PROVIDER", name)
        for field, value in PROVIDER_DEFAULTS[name].items():
            monkeypatch.setattr(settings, field, value)
    return use


def test_bedrock_builds_converse_clients_with_bedrock_ids(provider):
    provider("bedrock")
    small, reasoning = models.small_llm(max_tokens=10), models.reasoning_llm()
    assert isinstance(small, ChatBedrockConverse) and isinstance(reasoning, ChatBedrockConverse)
    assert small.model_id.startswith("eu.anthropic.")
    assert reasoning.additional_model_request_fields["thinking"] == {"type": "adaptive"}
    assert type(models.get_embeddings()).__name__ == "BedrockCohereEmbeddings"


def test_direct_builds_anthropic_clients_with_api_ids(provider, monkeypatch):
    provider("direct")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "test-key")
    small, reasoning = models.small_llm(max_tokens=10), models.reasoning_llm()
    assert isinstance(small, ChatAnthropic) and isinstance(reasoning, ChatAnthropic)
    assert small.model == "claude-haiku-4-5" and reasoning.model == "claude-sonnet-4-6"
    assert type(models.get_embeddings()).__name__ == "CohereEmbeddings"


def test_direct_without_anthropic_key_fails_clearly(provider, monkeypatch):
    provider("direct")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        models.small_llm(max_tokens=10)


def test_explicit_model_id_overrides_the_provider_default():
    s = Settings(MODEL_PROVIDER="direct", REASONING_MODEL_ID="claude-opus-5-5")
    assert s.REASONING_MODEL_ID == "claude-opus-5-5"
    assert s.MODEL_SMALL_ID == "claude-haiku-4-5"


def test_both_providers_use_the_same_embedding_size():
    # Même modèle et même dimension : changer de fournisseur ne demande pas de réindexer.
    assert {Settings(MODEL_PROVIDER=p).EMBEDDING_DIMENSIONS for p in PROVIDER_DEFAULTS} == {1024}
