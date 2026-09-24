"""Answer relevancy (méthode RAGAS) avec un LLM et des embeddings factices."""
import pytest

from evaluation.answer_relevancy import AnswerRelevancy, parse_generated
from conftest import FakeLLM


class BagOfWordsEmbeddings:
    """Embedding = sac de mots sur un petit vocabulaire : cosinus lisible dans les tests."""

    VOCAB = ["amd", "quick", "ratio", "2022", "boeing", "legal", "pepsico", "revenue"]

    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(w in t.lower()) for w in self.VOCAB] for t in texts]


def _scorer(llm_output, error=None):
    return AnswerRelevancy(llm=FakeLLM([llm_output], error=error), embeddings=BagOfWordsEmbeddings())


def test_on_topic_answer_scores_high():
    s = _scorer("EVASIVE: NON\nWhat is AMD quick ratio in 2022?\nAMD 2022 quick ratio?\nQuick ratio of AMD 2022?")
    assert s.score("What is the quick ratio of AMD in 2022?", "The quick ratio is 1.57 [3][4].") == 1.0


def test_answer_about_something_else_scores_low():
    s = _scorer("EVASIVE: NON\nWhat legal proceedings does Boeing face?\nBoeing legal matters?")
    assert s.score("What is the quick ratio of AMD in 2022?", "Boeing faces litigation [2].") == 0.0


def test_evasive_and_canned_refusals_score_zero():
    assert _scorer("EVASIVE: OUI\nq?").score("q", "Les passages ne permettent pas de conclure.") == 0.0
    s = _scorer("ne doit pas être appelé")
    assert s.score("q", "Cette information n'est pas disponible dans le document.") == 0.0
    assert s.llm.prompts == []  # refus figé : pas d'appel LLM


def test_technical_failures_are_missing_not_zero():
    assert _scorer("", error=RuntimeError("500")).score("q", "Revenue was $23.6B [1].") is None
    assert _scorer("EVASIVE: NON").score("q", "Revenue was $23.6B [1].") is None  # aucune question générée
    assert _scorer("x").score("q", "Une erreur est survenue lors de la génération de la réponse.") is None


def test_rate_limits_propagate_for_backoff():
    class RateLimited(Exception):
        status_code = 429

    with pytest.raises(RateLimited):
        _scorer("", error=RateLimited("429")).score("q", "Revenue was $23.6B [1].")


def test_citations_are_stripped_before_generation():
    s = _scorer("EVASIVE: NON\nAMD revenue 2022?")
    s.score("AMD revenue 2022?", "AMD revenue was $23.6B [1][2] in 2022 [5].")
    assert "[1]" not in s.llm.prompts[0] and "$23.6B" in s.llm.prompts[0]


def test_parse_generated_tolerates_formatting():
    evasive, qs = parse_generated("**EVASIVE:** NON\n1. What is AMD revenue?\n- Boeing legal issues?\n\nok", 3)
    assert evasive is False
    assert qs == ["What is AMD revenue?", "Boeing legal issues?"]
    assert parse_generated("EVASIVE: YES", 3) == (True, [])
