"""Sorties LLM « sales » : ce que le modèle renvoie vraiment, pas ce que le prompt demande.

Les autres tests stubbent les agents entiers ; ici on fait tourner le vrai code de parsing
derrière un FakeLLM, parce que c'est lui qui transforme une réponse mal formée en verdict
silencieusement faux."""
import pytest

from backend.agents.relevance_checker import RelevanceChecker, parse_relevance_label
from backend.config.settings import settings
from backend.retriever.multi_query import MultiQueryRetriever
from evaluation.llm_judge import FinanceBenchJudge
from conftest import FakeLLM, FakeRetriever, make_doc


@pytest.mark.parametrize("raw,expected", [
    ("CAN_ANSWER", "CAN_ANSWER"),
    ("CAN_ANSWER.", "CAN_ANSWER"),
    ("**PARTIAL**", "PARTIAL"),
    ('"NO_MATCH"', "NO_MATCH"),
    ("Label: can_answer", "CAN_ANSWER"),
    ("PARTIAL\nLes passages citent la dette sans le total.", "PARTIAL"),
    ("CAN ANSWER", "CAN_ANSWER"),
    ("NO-MATCH", "NO_MATCH"),
])
def test_relevance_label_survives_formatting(raw, expected):
    assert parse_relevance_label(raw) == expected


@pytest.mark.parametrize("raw", ["", "Je ne sais pas.", "PARTIALLY relevant", "CAN_ANSWERS"])
def test_relevance_label_rejects_non_labels(raw):
    assert parse_relevance_label(raw) is None


def _checker(contents=None, error=None):
    checker = RelevanceChecker.__new__(RelevanceChecker)
    checker.model = FakeLLM(contents, error=error)
    return checker


def test_relevance_checker_uses_the_parsed_label():
    """Avant : « **PARTIAL** » devenait NO_MATCH, donc un refus en mode sans outils."""
    assert _checker(["**PARTIAL**"]).check("q", [make_doc("x")]) == "PARTIAL"


def test_relevance_checker_falls_back_to_no_match():
    assert _checker(["bla"]).check("q", [make_doc("x")]) == "NO_MATCH"
    assert _checker().check("q", []) == "NO_MATCH"
    assert _checker(error=RuntimeError("500")).check("q", [make_doc("x")]) == "NO_MATCH"


def test_relevance_checker_lets_rate_limits_through():
    class RateLimited(Exception):
        status_code = 429

    with pytest.raises(RateLimited):
        _checker(error=RateLimited("429 Too Many Requests")).check("q", [make_doc("x")])


def test_relevance_checker_only_shows_top_k_passages():
    checker = _checker(["CAN_ANSWER"])
    checker.check("q", [make_doc(f"passage-{i}") for i in range(5)], k=2)
    prompt = checker.model.prompts[0]
    assert "passage-1" in prompt and "passage-2" not in prompt


@pytest.mark.parametrize("raw,verdict,faith", [
    ("VERDICT: CORRECT\nFAITHFULNESS: 5\nRAISON: ok", "CORRECT", 5.0),
    ("**VERDICT:** CORRECT\n**FAITHFULNESS:** 4\n**RAISON:** ok", "CORRECT", 4.0),
    ("VERDICT**: REFUSAL\nFAITHFULNESS**: 4.5\nRAISON**: ok", "REFUSAL", 4.5),
    ("VERDICT: [INCORRECT]\nFAITHFULNESS: [2]\nRAISON: chiffre faux", "INCORRECT", 2.0),
    ("verdict: correct\nfaithfulness: 9\nraison: ok", "CORRECT", 5.0),  # borné à 5
])
def test_judge_parse_tolerates_markdown(raw, verdict, faith):
    parsed = FinanceBenchJudge()._parse(raw)
    assert (parsed.verdict, parsed.faithfulness) == (verdict, faith)
    assert parsed.reason != "Impossible de parser la réponse du juge"


def test_judge_parse_defaults_are_pessimistic():
    """Réponse illisible : compter INCORRECT (jamais CORRECT par accident) et le dire."""
    parsed = FinanceBenchJudge()._parse("Je pense que c'est bon.")
    assert parsed.verdict == "INCORRECT"
    assert parsed.reason == "Impossible de parser la réponse du juge"


def test_judge_reads_the_llm_verdict_end_to_end():
    judge = FinanceBenchJudge(llm=FakeLLM(["**VERDICT:** CORRECT\n**FAITHFULNESS:** 5\n**RAISON:** exact"]))
    v = judge.evaluate("q", "$9,068", "EBITDA less capex: $9,068 million [3].", "ctx")
    assert v.verdict == "CORRECT" and v.reason == "exact"


def test_multi_query_parses_reformulations_and_keeps_original_first(monkeypatch):
    monkeypatch.setattr(settings, "MULTI_QUERY_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_COUNT", 3)
    llm = FakeLLM(["AMD total revenue 2022\n\n- puce ignorée\nAMD net revenue by segment\nAMD sales growth\nen trop"])
    mq = MultiQueryRetriever(FakeRetriever(), llm=llm)
    assert mq._generate_queries("revenus AMD ?") == [
        "revenus AMD ?", "AMD total revenue 2022", "AMD net revenue by segment", "AMD sales growth",
    ]


def test_multi_query_falls_back_to_original_on_llm_error(monkeypatch):
    monkeypatch.setattr(settings, "MULTI_QUERY_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_COUNT", 3)
    mq = MultiQueryRetriever(FakeRetriever(), llm=FakeLLM(error=RuntimeError("timeout")))
    assert mq._generate_queries("q") == ["q"]
