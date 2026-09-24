"""La logique de décision des garde-fous de régression (evaluation/financebench/regression.py).

Les runs eux-mêmes appellent les API ; ce qui est testé ici, c'est ce qui décide « régression
ou pas » — un garde-fou qui laisse passer une régression est pire que pas de garde-fou."""
import json
from pathlib import Path

from backend.config.settings import settings
from evaluation.financebench.regression import (
    SENTINELS_PATH,
    DegradationWatch,
    check_sentinel,
    compare_retrieval,
    per_question,
)
from evaluation.utils import load_dataset
from conftest import FakeRetriever, make_doc


def _q(rank, hit=None):
    hit = (1.0 if rank is not None and rank <= 10 else 0.0) if hit is None else hit
    return {"gold_rank": rank, "page_hit@10": hit, "page_recall@10": hit, "mrr@10": 1 / rank if rank else 0.0}


def test_identical_runs_pass():
    base = {"a": _q(1), "b": _q(15), "c": _q(None)}
    assert compare_retrieval(base, dict(base))["ok"]


def test_losing_one_question_fails_even_if_the_mean_is_unchanged():
    """Une perte compensée par un gain laisse page_hit@10 identique : la moyenne ne voit rien."""
    base = {"a": _q(3), "b": _q(15)}
    cur = {"a": _q(12), "b": _q(4)}
    report = compare_retrieval(base, cur)
    assert report["metrics"]["page_hit@10"]["delta"] == 0
    assert not report["ok"]
    assert report["lost"] == [{"id": "a", "before": 3, "after": 12}]
    assert report["gained"] == [{"id": "b", "before": 15, "after": 4}]


def test_tolerances_are_respected():
    base = {f"q{i}": _q(1) for i in range(20)}
    cur = dict(base, q0=_q(None))
    assert not compare_retrieval(base, cur)["ok"]
    assert compare_retrieval(base, cur, max_lost=1)["ok"]  # 1 perte sur 20 = -5 pts : à la limite
    cur2 = dict(cur, q1=_q(None))
    assert not compare_retrieval(base, cur2, max_lost=2)["ok"]  # -10 pts > 5 pts tolérés


def test_moves_inside_the_top_k_are_reported_not_failed():
    report = compare_retrieval({"a": _q(2)}, {"a": _q(8)})
    assert report["ok"] and report["moved"] == [{"id": "a", "before": 2, "after": 8}]


def test_missing_questions_fail_the_gate():
    report = compare_retrieval({"a": _q(1), "b": _q(2)}, {"a": _q(1)})
    assert not report["ok"] and report["missing"] == ["b"]


def test_per_question_keeps_one_row_per_id():
    rows = [{"id": "a", "mode": "baseline", "gold_rank": 3}, {"id": "a", "mode": "agentic", "gold_rank": 3}]
    assert list(per_question(rows)) == ["a"]


def test_sentinel_checks_verdict_evidence_and_tools():
    expect = {"verdict": "CORRECT", "evidence_seen": True, "min_tool_calls": 1}
    good = {"verdict": "CORRECT", "evidence_seen": True, "tool_calls": 2}
    assert check_sentinel(good, expect) == []
    assert len(check_sentinel({"verdict": "INCORRECT", "evidence_seen": False, "tool_calls": 0}, expect)) == 3
    assert check_sentinel({"failed": True, "error": "timeout"}, expect)[0].startswith("échec technique")


def test_sentinels_exist_in_the_dataset_and_are_explained():
    ids = {ex["id"] for ex in load_dataset(str(Path(SENTINELS_PATH).parents[1] / "dataset.jsonl"))}
    sentinels = json.loads(Path(SENTINELS_PATH).read_text(encoding="utf-8"))["sentinels"]
    assert len(sentinels) >= 8
    for s in sentinels:
        assert s["id"] in ids
        assert s.get("why")


def test_degradation_watch_flags_silent_rerank_fallback(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    reranked = make_doc("a")
    reranked.metadata["rerank_score"] = 0.9
    ok = DegradationWatch(FakeRetriever(default=[reranked]))
    ok.invoke("q")
    assert ok.problems(1) == []

    degraded = DegradationWatch(FakeRetriever(default=[make_doc("a")]))
    degraded.invoke("q")
    assert "rerank" in degraded.problems(1)[0]
    assert degraded.calls == ["q"]  # les autres attributs passent au retriever enveloppé


def test_other_document_sets_never_overwrite_the_published_dataset():
    from evaluation.financebench.prepare import DEFAULT_DOCS, dataset_path_for, extended_docs

    assert dataset_path_for(list(reversed(DEFAULT_DOCS))).name == "dataset.jsonl"
    assert dataset_path_for(["AMD_2022_10K"]).name == "dataset_custom.jsonl"
    questions = [{"doc_name": d} for d in ["A"] * 3 + ["B"] * 2 + ["C"] * 4]
    assert extended_docs(questions) == ["A", "C"]
    assert dataset_path_for(["A", "C"], "extended").name == "dataset_extended.jsonl"


def test_non_default_dataset_cannot_overwrite_published_outputs(tmp_path):
    import pytest
    from evaluation.financebench.run_financebench_eval import guard_partial_overwrite

    default = str(tmp_path / "outputs")
    with pytest.raises(SystemExit):
        guard_partial_overwrite(default, default, ["--dataset dataset_extended.jsonl"], force=False)
    guard_partial_overwrite(str(tmp_path / "ailleurs"), default, ["--dataset dataset_extended.jsonl"], force=False)
