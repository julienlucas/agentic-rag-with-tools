# Métriques d'évaluation partagées : retrieval (recall@k, precision@k, MRR, nDCG) et réponse (F1,
# context_hit). Extraites de l'ancien harness `evaluation/run_eval.py` (supprimé avec le
# jeu interne) pour que l'évaluation FinanceBench garde exactement le même calcul.

import re
import unicodedata
import math
from typing import List, Optional



def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> List[str]:
    return _normalize(text).split()


def _f1_score(pred: str, ref: str) -> float:
    pred_toks = _tokens(pred)
    ref_toks = _tokens(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    common = {}
    for tok in pred_toks:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in ref_toks:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    precision = overlap / max(len(pred_toks), 1)
    recall = overlap / max(len(ref_toks), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _context_hits(docs, expected_answer: str, keywords: Optional[List[str]]) -> bool:
    context = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    if expected_answer:
        if _normalize(expected_answer) in _normalize(context):
            return True
    if keywords:
        norm_ctx = _normalize(context)
        return any(_normalize(k) in norm_ctx for k in keywords if k)
    return False


def _normalize_list(values: Optional[List[str]]) -> List[str]:
    return [_normalize(v) for v in (values or []) if v]


def _token_overlap_score(gold_text: str, doc_text: str) -> float:
    """Calcule le ratio de tokens du gold présents dans le doc."""
    gold_toks = _tokens(gold_text)
    if not gold_toks:
        return 0.0
    doc_toks_set = set(_tokens(doc_text))
    matched = sum(1 for t in gold_toks if t in doc_toks_set)
    return matched / len(gold_toks)


def _doc_relevance_flags(docs, gold_passages: Optional[List[str]], fuzzy_threshold: float = 0.6) -> List[int]:
    """
    Détermine la pertinence de chaque doc par rapport aux gold passages.
    Utilise un matching hybride : substring exact OU token overlap >= seuil.
    """
    if not gold_passages:
        return []
    gold = _normalize_list(gold_passages)
    flags = []
    for doc in docs:
        content = _normalize(getattr(doc, "page_content", ""))
        # Match exact (substring) — rapide
        if any(g in content for g in gold):
            flags.append(1)
            continue
        # Match fuzzy (token overlap) — rattrape les reformulations et coupures
        if any(_token_overlap_score(g, content) >= fuzzy_threshold for g in gold):
            flags.append(1)
            continue
        flags.append(0)
    return flags


def _recall_at_k(flags: List[int], k: int) -> Optional[float]:
    if not flags:
        return None
    total_relevant = sum(flags)
    if total_relevant == 0:
        return 0.0
    return min(sum(flags[:k]) / total_relevant, 1.0)


def _precision_at_k(flags: List[int], k: int) -> Optional[float]:
    """
    Part des k premiers passages qui sont pertinents (dénominateur k, convention standard :
    renvoyer moins de k passages ne gonfle pas la précision).

    Sur FinanceBench, une question n'a souvent qu'une ou deux preuves : precision@10 est
    plafonnée bien en dessous de 1 par construction. Elle se lit en évolution d'un run à
    l'autre (moins de bruit dans le contexte), pas en valeur absolue.
    """
    if not flags or k <= 0:
        return None
    return sum(flags[:k]) / k


def _mrr_at_k(flags: List[int], k: int) -> Optional[float]:
    if not flags:
        return None
    for idx, rel in enumerate(flags[:k], start=1):
        if rel:
            return 1.0 / idx
    return 0.0


def _ndcg_at_k(flags: List[int], k: int) -> Optional[float]:
    if not flags:
        return None
    dcg = 0.0
    for i, rel in enumerate(flags[:k], start=1):
        if rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sorted(flags, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k], start=1):
        if rel:
            idcg += 1.0 / math.log2(i + 1)
    if idcg == 0:
        return 0.0
    return dcg / idcg
