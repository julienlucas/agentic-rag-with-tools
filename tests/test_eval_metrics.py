import math

from evaluation.metrics import _f1_score, _mrr_at_k, _ndcg_at_k, _precision_at_k, _recall_at_k


def test_recall_at_k():
    assert _recall_at_k([0, 1, 0, 1], 2) == 0.5
    assert _recall_at_k([1, 1, 0], 3) == 1.0
    assert _recall_at_k([0, 0, 0], 3) == 0.0
    assert _recall_at_k([], 3) is None


def test_precision_at_k():
    assert _precision_at_k([1, 0, 1, 0], 2) == 0.5
    assert _precision_at_k([1, 1, 0, 0], 4) == 0.5
    assert _precision_at_k([1], 5) == 0.2  # moins de k passages : le dénominateur reste k
    assert _precision_at_k([], 3) is None


def test_mrr_at_k():
    assert _mrr_at_k([1, 0, 0], 3) == 1.0
    assert _mrr_at_k([0, 1, 0], 3) == 0.5
    assert _mrr_at_k([0, 0, 1], 2) == 0.0  # hors fenêtre k


def test_ndcg_at_k():
    assert _ndcg_at_k([1, 0, 0], 3) == 1.0
    assert math.isclose(_ndcg_at_k([0, 1, 0], 3), 1 / math.log2(3), rel_tol=1e-9)
    assert _ndcg_at_k([0, 0, 0], 3) == 0.0


def test_token_f1():
    assert _f1_score("Revenue was 10", "revenue was 10") == 1.0
    assert _f1_score("alpha beta", "gamma delta") == 0.0
    assert math.isclose(_f1_score("the revenue was 10", "revenue 10"), 2 / 3, rel_tol=1e-9)
