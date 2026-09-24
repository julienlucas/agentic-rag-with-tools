"""Agrégat et tableau de fin de run FinanceBench : chaque métrique mesurée doit y remonter."""
import io

from evaluation.financebench import run_financebench_eval as R
from conftest import make_doc


def test_retrieval_metrics_include_precision():
    docs = [make_doc("x", source="AMD", page=p) for p in (3, 40, 41, 3)]
    for d in docs:
        d.metadata["doc_name"] = "AMD"
    ex = {"gold_pages": [["AMD", 3]], "gold_passages": []}
    m = R.compute_retrieval_metrics(docs, ex, [2, 4], page_tolerance=0)
    assert m["page_precision@2"] == 0.5 and m["page_precision@4"] == 0.5
    assert m["gold_rank"] == 1


def _row(relevancy, verdict="CORRECT"):
    return {
        "answer_f1": 0.1, "context_hit": True, "generation_sec": 1.0, "retrieval_sec": 0.5,
        "evidence_seen": True, "verdict": verdict, "judge_faithfulness": 5.0,
        "answer_relevancy": relevancy, "precision@5": 0.2, "page_precision@5": 0.4,
    }


def test_aggregate_and_report_show_new_metrics(monkeypatch):
    agg = R.aggregate([_row(0.9), _row(0.7), _row(None, "REFUSAL")], [5])
    assert agg["mean_answer_relevancy"] == 0.8 and agg["answer_relevancy_count"] == 2
    assert agg["retrieval"]["precision@5"] == 0.2 and agg["retrieval"]["page_precision@5"] == 0.4

    out = io.StringIO()
    monkeypatch.setattr(R, "_REAL_STDOUT", out)
    R.print_report({"agentic": agg, "n_questions": 3}, ["agentic"], [5])
    text = out.getvalue()
    assert "Answer relevancy (0-1)" in text and "0.8" in text
    assert "page_precision@5" in text and "precision@5 (texte)" in text
    assert "Faithfulness moyenne /5" in text
