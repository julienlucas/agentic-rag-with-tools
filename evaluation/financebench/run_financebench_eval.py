# Évaluation du RAG agentique sur FinanceBench (Patronus AI).
#
# Protocole FinanceBench : accuracy / refusal_rate / hallucination_rate, jugés par LLM,
# plus la faithfulness (note du juge) et l'answer relevancy (méthode RAGAS).
# Métriques retrieval exactes grâce aux pages annotées (page_hit@k, page_recall@k,
# page_precision@k), plus les métriques historiques du projet (recall/precision/mrr/ndcg,
# F1, context_hit).
#
# Prérequis: uv run python evaluation/financebench/prepare.py
#
# Usage:
#   uv run python evaluation/financebench/run_financebench_eval.py --mode both
#   uv run python evaluation/financebench/run_financebench_eval.py --max-items 3 --no-judge --out-dir /tmp/fb-essai

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.agents.research_agent import ResearchAgent
from backend.config.settings import settings
from backend.agents.workflow import AgentState, AgentWorkflow, effective_top_k
from backend.retriever.page_store import PageStore
from evaluation.financebench.prepare import load_cached_chunks, load_cached_pages, store_dir_for
from evaluation.answer_relevancy import AnswerRelevancy
from evaluation.llm_judge import FinanceBenchJudge, aggregate_financebench_verdicts
from evaluation.financebench.cost import compute_cost, format_cost
from evaluation.metrics import (
    _context_hits,
    _doc_relevance_flags,
    _f1_score,
    _mrr_at_k,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
)
from evaluation.utils import (
    build_retriever_from_chunks,
    call_with_backoff,
    is_rate_limit,
    load_dataset,
    log_to_langsmith,
    root_cause,
)

HERE = Path(__file__).resolve().parent

# stdout réel, conservé avant que le pipeline (très bavard) ne soit mis en sourdine.
_REAL_STDOUT = sys.stdout
_PRINT_LOCK = threading.Lock()


def _log(msg: str):
    with _PRINT_LOCK:
        print(f"[financebench] {msg}", file=_REAL_STDOUT, flush=True)


# ---------------------------------------------------------------------------
# Métriques de retrieval basées sur les pages annotées
# ---------------------------------------------------------------------------

def _page_flags(docs, gold_pages: List, tolerance: int = 1) -> List[int]:
    """
    1 si le chunk provient d'une page annotée comme portant la preuve, 0 sinon.

    La tolérance absorbe un éventuel décalage entre la pagination du PDF (annotations
    FinanceBench, zero-indexed) et celle de l'OCR Mistral.
    """
    if not gold_pages:
        return []
    gold = {(str(d), int(p)) for d, p in gold_pages}
    flags = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        doc_name = str(meta.get("doc_name") or meta.get("source") or "")
        page = meta.get("page")
        hit = 0
        if page is not None:
            for gold_doc, gold_page in gold:
                if doc_name == gold_doc and abs(int(page) - gold_page) <= tolerance:
                    hit = 1
                    break
        flags.append(hit)
    return flags


def _page_hit_at_k(flags: List[int], k: int) -> Optional[float]:
    """Au moins un chunk du top-k vient-il d'une page de preuve ?"""
    if not flags:
        return None
    return 1.0 if any(flags[:k]) else 0.0


def _page_recall_at_k(docs, gold_pages: List, k: int, tolerance: int = 1) -> Optional[float]:
    """Proportion des pages de preuve distinctes couvertes par le top-k."""
    if not gold_pages:
        return None
    gold = {(str(d), int(p)) for d, p in gold_pages}
    covered = set()
    for doc in docs[:k]:
        meta = getattr(doc, "metadata", {}) or {}
        doc_name = str(meta.get("doc_name") or meta.get("source") or "")
        page = meta.get("page")
        if page is None:
            continue
        for gold_doc, gold_page in gold:
            if doc_name == gold_doc and abs(int(page) - gold_page) <= tolerance:
                covered.add((gold_doc, gold_page))
    return len(covered) / len(gold)


def compute_retrieval_metrics(docs, example: Dict, k_values: List[int], page_tolerance: int) -> Dict:
    """
    Métriques de retrieval d'une question, partagées par les modes — et par le garde-fou de
    régression (regression.py), qui doit mesurer exactement la même chose que l'éval.
    """
    gold_passages = example.get("gold_passages", [])
    gold_pages = example.get("gold_pages", [])
    text_flags = _doc_relevance_flags(docs, gold_passages)
    page_flags = _page_flags(docs, gold_pages, page_tolerance)
    metrics = {"n_docs": len(docs)}
    for k in k_values:
        metrics[f"recall@{k}"] = _recall_at_k(text_flags, k)
        metrics[f"precision@{k}"] = _precision_at_k(text_flags, k)
        metrics[f"mrr@{k}"] = _mrr_at_k(text_flags, k)
        metrics[f"ndcg@{k}"] = _ndcg_at_k(text_flags, k)
        metrics[f"page_hit@{k}"] = _page_hit_at_k(page_flags, k)
        metrics[f"page_recall@{k}"] = _page_recall_at_k(docs, gold_pages, k, page_tolerance)
        metrics[f"page_precision@{k}"] = _precision_at_k(page_flags, k)

    # Traçabilité : les 20 (document, page) récupérés + rang de la première page de preuve.
    metrics["retrieved"] = [
        [str((d.metadata or {}).get("doc_name") or (d.metadata or {}).get("source") or ""),
         (d.metadata or {}).get("page")]
        for d in docs[:20]
    ]
    metrics["gold_rank"] = next((i for i, f in enumerate(page_flags, start=1) if f), None)
    return metrics


# ---------------------------------------------------------------------------
# Évaluation d'une question
# ---------------------------------------------------------------------------

def evaluate_example(
    example: Dict,
    retriever,
    modes: List[str],
    workflow: AgentWorkflow,
    researcher: ResearchAgent,
    judge: Optional[FinanceBenchJudge],
    k_values: List[int],
    page_tolerance: int,
    page_store: Optional[PageStore] = None,
    relevancy: Optional[AnswerRelevancy] = None,
) -> List[Dict]:
    """Récupère une seule fois, puis génère une réponse par mode évalué."""
    question = example["question"].strip()
    expected = example.get("expected_answer", "").strip()
    gold_pages = example.get("gold_pages", [])

    t0 = time.time()
    docs = retriever.invoke(question)
    retrieval_metrics = compute_retrieval_metrics(docs, example, k_values, page_tolerance)
    retrieval_metrics["retrieval_sec"] = round(time.time() - t0, 2)

    # Documents transmis au LLM (même top-k que workflow._research_step).
    top_k = settings.RESEARCH_TOP_K
    top_docs = docs[:top_k]

    rows = []
    for mode in modes:
        row = {
            "id": example.get("id"),
            "doc_name": example.get("doc_name"),
            "question": question,
            "expected_answer": expected,
            "question_type": example.get("question_type"),
            "question_reasoning": example.get("question_reasoning"),
            "mode": mode,
            **retrieval_metrics,
        }

        # Chaque mode est isolé : l'échec de l'un ne doit pas faire perdre le résultat
        # de l'autre ni, surtout, retirer silencieusement la question de l'échantillon.
        t1 = time.time()
        try:
            if mode == "baseline":
                # RAG simple : même retrieval, une seule génération, sans boucle agentique.
                answer = call_with_backoff(
                    lambda: researcher.generate(question, top_docs)["draft_answer"],
                    f"la génération baseline ({example.get('id')})", log=_log,
                )
            else:
                # On invoque directement le graphe compilé avec les documents déjà récupérés :
                # full_pipeline() relancerait un retrieval complet (~2x la latence).
                state = AgentState(
                    question=question,
                    documents=docs,
                    draft_answer="",
                    verification_report="",
                    is_relevant=False,
                    retriever=retriever,
                    page_store=page_store,
                    corrective_rounds=0,
                    tool_calls=0,
                )
                final_state = call_with_backoff(
                    lambda: workflow.compiled_workflow.invoke(state),
                    f"la génération agentic ({example.get('id')})", log=_log,
                )
                answer = final_state["draft_answer"]
        except Exception as e:
            # ResearchAgent.generate masque l'erreur d'origine derrière un RuntimeError
            # générique : on déroule la chaîne des causes pour savoir ce qui s'est passé.
            row.update({
                "failed": True,
                "error": root_cause(e),
                "rate_limited": is_rate_limit(e),
                "answer": "",
                "generation_sec": round(time.time() - t1, 2),
            })
            if judge:
                row["verdict"] = "ERROR"
            rows.append(row)
            continue

        # Documents effectivement transmis au LLM pour CE mode. En agentic, la
        # recherche corrective a pu les changer : c'est précisément ce qu'on mesure.
        if mode == "agentic":
            llm_docs = (final_state.get("documents") or docs)[:effective_top_k(final_state)]
            row["corrective_rounds"] = final_state.get("corrective_rounds", 0)
            row["corrective_queries"] = final_state.get("corrective_queries", [])
            row["tool_calls"] = final_state.get("tool_calls", 0)
            row["pages_read"] = sorted({
                f"{(d.metadata or {}).get('doc_name')} p.{(d.metadata or {}).get('page')}"
                for d in llm_docs if (d.metadata or {}).get("origin") == "read_page"
            })
            row["relevance"] = final_state.get("relevance", "")
        else:
            llm_docs = top_docs
        llm_flags = _page_flags(llm_docs, gold_pages, page_tolerance)
        row["evidence_seen"] = bool(any(llm_flags)) if gold_pages else None

        row["answer"] = answer
        row["generation_sec"] = round(time.time() - t1, 2)
        row["context_hit"] = _context_hits(llm_docs, expected, example.get("answer_keywords"))
        row["answer_f1"] = _f1_score(answer, expected) if expected else 0.0

        context = "\n\n".join(getattr(d, "page_content", "") for d in llm_docs)
        if judge:
            try:
                verdict = call_with_backoff(
                    lambda: judge.evaluate(
                        question=question,
                        expected_answer=expected,
                        generated_answer=answer,
                        context=context,
                        justification=example.get("justification", ""),
                    ),
                    f"le juge ({example.get('id')})", log=_log,
                )
                row["verdict"] = verdict.verdict
                row["judge_faithfulness"] = verdict.faithfulness
                row["judge_reason"] = verdict.reason
            except Exception as e:
                # La réponse est valide, seul le juge a échoué : on garde la réponse.
                row["verdict"] = "ERROR"
                row["judge_reason"] = f"juge indisponible: {root_cause(e)}"

        if relevancy:
            try:
                row["answer_relevancy"] = call_with_backoff(
                    lambda: relevancy.score(question, answer),
                    f"l'answer relevancy ({example.get('id')})", log=_log,
                )
            except Exception:
                row["answer_relevancy"] = None  # mesure manquante, la réponse reste valide

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------

def aggregate(results: List[Dict], k_values: List[int]) -> Dict:
    if not results:
        return {}

    failed = [r for r in results if r.get("failed")]
    ok = [r for r in results if not r.get("failed")]
    if not ok:
        return {"count": 0, "failed": len(failed), "attempted": len(results)}

    results = ok
    base = {
        "count": len(results),
        "failed": len(failed),
        "rate_limited": sum(1 for r in failed if r.get("rate_limited")),
        "attempted": len(ok) + len(failed),
        "mean_f1": round(mean(r["answer_f1"] for r in results), 4),
        "context_hit_rate": round(sum(1 for r in results if r["context_hit"]) / len(results), 4),
        "mean_generation_sec": round(mean(r["generation_sec"] for r in results), 2),
        "mean_retrieval_sec": round(mean(r["retrieval_sec"] for r in results), 2),
    }
    seen = [r["evidence_seen"] for r in results if r.get("evidence_seen") is not None]
    if seen:
        base["evidence_seen_rate"] = round(sum(seen) / len(seen), 4)
    relevancy = [r["answer_relevancy"] for r in results if r.get("answer_relevancy") is not None]
    if relevancy:
        base["mean_answer_relevancy"] = round(mean(relevancy), 4)
        base["answer_relevancy_count"] = len(relevancy)
    corrective = [r.get("corrective_rounds") for r in results if r.get("corrective_rounds") is not None]
    if corrective:
        base["corrective_rate"] = round(sum(1 for c in corrective if c > 0) / len(corrective), 4)
        corrected = [r for r in results if r.get("corrective_rounds")]
        if corrected:
            base["mean_tool_calls_when_corrected"] = round(
                mean(r.get("tool_calls", 0) for r in corrected), 2)
            base["pages_read_rate_when_corrected"] = round(
                sum(1 for r in corrected if r.get("pages_read")) / len(corrected), 4)

    retrieval = {}
    for k in k_values:
        for metric in (f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}",
                       f"page_hit@{k}", f"page_recall@{k}", f"page_precision@{k}"):
            values = [r.get(metric) for r in results if r.get(metric) is not None]
            if values:
                retrieval[metric] = round(mean(values), 4)
    base["retrieval"] = retrieval

    verdicts = [r for r in results if r.get("verdict")]
    if verdicts:
        from evaluation.llm_judge import FinanceBenchVerdict
        base["financebench"] = aggregate_financebench_verdicts([
            FinanceBenchVerdict(r["verdict"], r.get("judge_faithfulness", 0.0), "")
            for r in verdicts
        ])

    return base


def breakdown(results: List[Dict], field: str, k_values: List[int]) -> Dict:
    """Ventile accuracy / page_hit par type ou par raisonnement de question."""
    groups = defaultdict(list)
    for row in results:
        groups[row.get(field) or "unknown"].append(row)

    k = max(k_values) if k_values else 10
    page_key = f"page_hit@{k}"

    out = {}
    for key, rows in sorted(groups.items()):
        verdicts = [r["verdict"] for r in rows if r.get("verdict")]
        page_hits = [r.get(page_key) for r in rows if r.get(page_key) is not None]
        out[key] = {
            "count": len(rows),
            "accuracy": round(sum(1 for v in verdicts if v == "CORRECT") / len(verdicts), 4) if verdicts else None,
            "hallucination_rate": round(sum(1 for v in verdicts if v == "INCORRECT") / len(verdicts), 4) if verdicts else None,
            "refusal_rate": round(sum(1 for v in verdicts if v == "REFUSAL") / len(verdicts), 4) if verdicts else None,
            page_key: round(mean(page_hits), 4) if page_hits else None,
            "mean_f1": round(mean(r["answer_f1"] for r in rows), 4),
        }
    return out


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def print_report(summary: Dict, modes: List[str], k_values: List[int] = None, n_protocol: int = 0):
    """Tableau de synthèse lisible, dans le format des chiffres publics de Mistral."""
    lines = ["", "=" * 78, "RÉSULTATS — FinanceBench (protocole Patronus AI)", "=" * 78]

    # Un bloc entier de tirets n'apprend rien à personne : quand les verdicts manquent,
    # la première chose à dire est POURQUOI, et quelle commande donne le vrai résultat.
    n_run = summary.get("n_questions") or 0
    has_verdicts = any((summary.get(m) or {}).get("financebench") for m in modes)
    if not has_verdicts:
        lines += [
            "",
            "⚠️  AUCUN VERDICT — les métriques du protocole (accuracy, hallucinations, refus)",
            "    sont vides parce que le juge LLM n'a pas tourné (--no-judge, ou aucune",
            "    réponse jugeable). Ce run ne mesure que le retrieval et la latence.",
            "",
            "    Pour le résultat complet :",
            "      uv run python evaluation/financebench/run_financebench_eval.py --mode both",
            "=" * 78,
        ]
    elif n_run and n_protocol and n_run < n_protocol:
        lines += [
            "",
            f"⚠️  RUN PARTIEL — {n_run} question(s) sur les {n_protocol} du protocole. Les pourcentages",
            "    ci-dessous ne sont pas comparables aux chiffres publiés du projet.",
            "=" * 78,
        ]

    header = f"{'Métrique':<34}" + "".join(f"{m:>14}" for m in modes)
    lines.append(header)
    lines.append("-" * 78)

    def row(label, getter):
        cells = ""
        for mode in modes:
            block = summary.get(mode) or {}
            cells += f"{getter(block):>14}"
        lines.append(f"{label:<34}{cells}")

    k_values = k_values or [5, 10]
    k_small, k_large = min(k_values), max(k_values)
    fb = lambda b: b.get("financebench") or {}

    def _count(block, key):
        """Le comptage brut : 16/21 se discute, « 76,2 % » se subit."""
        counts = fb(block).get("counts") or {}
        k, n = counts.get(key), fb(block).get("count")
        return "—" if k is None or not n else f"{k}/{n}"

    def _ci(block, key):
        bounds = fb(block).get(f"{key}_ci95")
        return "—" if not bounds else f"[{bounds[0] * 100:.0f}-{bounds[1] * 100:.0f}%]"

    if has_verdicts:
        row("Accuracy (CORRECT)", lambda b: _pct(fb(b).get("accuracy")))
        row("  questions", lambda b: _count(b, "correct"))
        row("  IC95 %", lambda b: _ci(b, "accuracy"))
        row("Hallucinations (INCORRECT)", lambda b: _pct(fb(b).get("hallucination_rate")))
        row("  questions", lambda b: _count(b, "hallucination"))
        row("  IC95 %", lambda b: _ci(b, "hallucination_rate"))
        row("Refus (REFUSAL)", lambda b: _pct(fb(b).get("refusal_rate")))
        row("  questions", lambda b: _count(b, "refusal"))
        row("Faithfulness moyenne /5", lambda b: str(fb(b).get("mean_faithfulness") or "—"))
        lines.append("-" * 78)
    row("Answer relevancy (0-1)", lambda b: str(b.get("mean_answer_relevancy") or "—"))
    row("Preuve transmise au LLM", lambda b: _pct(b.get("evidence_seen_rate")))
    row("Recherche corrective déclenchée", lambda b: _pct(b.get("corrective_rate")))
    row("  appels d'outils (moy., si corrigée)", lambda b: str(b.get("mean_tool_calls_when_corrected", "—")))
    row("  page entière lue (si corrigée)", lambda b: _pct(b.get("pages_read_rate_when_corrected")))
    lines.append("-" * 78)
    row(f"page_hit@{k_small} (retrieval exact)", lambda b: _pct((b.get("retrieval") or {}).get(f"page_hit@{k_small}")))
    row(f"page_hit@{k_large}", lambda b: _pct((b.get("retrieval") or {}).get(f"page_hit@{k_large}")))
    row(f"page_recall@{k_large}", lambda b: _pct((b.get("retrieval") or {}).get(f"page_recall@{k_large}")))
    row(f"page_precision@{k_small}", lambda b: _pct((b.get("retrieval") or {}).get(f"page_precision@{k_small}")))
    row(f"recall@{k_large} (texte)", lambda b: _pct((b.get("retrieval") or {}).get(f"recall@{k_large}")))
    row(f"precision@{k_small} (texte)", lambda b: _pct((b.get("retrieval") or {}).get(f"precision@{k_small}")))
    row("context_hit_rate", lambda b: _pct(b.get("context_hit_rate")))
    row("mean_f1", lambda b: _pct(b.get("mean_f1")))
    lines.append("-" * 78)
    row("Latence retrieval (s/question)", lambda b: str(b.get("mean_retrieval_sec") or "—"))
    row("Latence génération (s/question)", lambda b: str(b.get("mean_generation_sec") or "—"))
    row("Questions évaluées", lambda b: f"{b.get('count') or 0}/{b.get('attempted') or 0}")
    row("Questions en échec", lambda b: str(b.get("failed") or 0))

    lines.append("=" * 78)
    lines.append(f"Durée totale: {summary.get('elapsed_sec')}s")
    if summary.get("cost"):
        lines.append(format_cost(summary["cost"]))

    expected = summary.get("n_questions") or 0
    worst = max((summary.get(m) or {}).get("failed", 0) for m in modes) if modes else 0
    rl = max((summary.get(m) or {}).get("rate_limited", 0) for m in modes) if modes else 0
    if worst:
        lines.append("")
        lines.append(f"⚠️  {worst}/{expected} questions ont ÉCHOUÉ — les pourcentages ci-dessus")
        lines.append("    ne portent que sur les questions abouties : ils ne sont PAS comparables")
        lines.append("    d'un run à l'autre tant que l'échantillon n'est pas complet.")
        if rl:
            lines.append(f"    Dont {rl} par rate limit. Relancez avec --workers 1.")
        lines.append("    Détail: evaluation/financebench/outputs/financebench_errors.json")
    if summary.get("truncated"):
        lines.append("⚠️  Run interrompu par --time-budget : résultats partiels.")

    # Garde-fou : c'est exactement la lecture abusive que ce tableau invite à faire.
    if len(modes) == 2:
        b_n = ((summary.get("baseline") or {}).get("financebench") or {}).get("counts") or {}
        a_n = ((summary.get("agentic") or {}).get("financebench") or {}).get("counts") or {}
        if b_n.get("correct") is not None and a_n.get("correct") is not None:
            gap = a_n["correct"] - b_n["correct"]
            lines.append("")
            lines.append(f"ℹ️  Écart agentic − baseline : {gap:+d} question(s) correcte(s) sur "
                         f"{fb(summary.get('agentic') or {}).get('count')}.")
            if abs(gap) <= 2:
                lines.append("    Les IC95 se recouvrent largement : à cette taille d'échantillon,")
                lines.append("    ce n'est pas une amélioration démontrable. Ne pas le citer comme telle.")
    lines.append("")
    print("\n".join(lines), file=_REAL_STDOUT, flush=True)


# ---------------------------------------------------------------------------

def guard_partial_overwrite(out_dir: str, default_dir: str, partial_reasons: list, force: bool):
    """
    Un run partiel ne doit jamais écraser les sorties versionnées.

    Ces fichiers portent les chiffres cités dans le README et l'étude de cas. Un
    `--max-items 3 --no-judge` lancé pour vérifier que la chaîne tourne les remplaçait
    silencieusement par 3 questions sans verdict — et le prochain `git commit -a` publiait
    ça comme le résultat officiel.
    """
    if force or not partial_reasons:
        return
    if os.path.abspath(out_dir) != os.path.abspath(default_dir):
        return
    raise SystemExit(
        "Refus d'écrire un run partiel (" + ", ".join(partial_reasons) + ") dans les sorties\n"
        f"versionnées ({default_dir}).\n"
        "  --out-dir /tmp/eval-essai   pour un essai jetable\n"
        "  --force-overwrite           si vous voulez vraiment remplacer les chiffres publiés"
    )


def main():
    parser = argparse.ArgumentParser(description="Évaluation FinanceBench du RAG agentique")
    parser.add_argument("--dataset", default=str(HERE / "dataset.jsonl"))
    parser.add_argument("--docs", default="", help="Sous-ensemble de documents (défaut: ceux du dataset)")
    parser.add_argument("--mode", default="both", choices=["baseline", "agentic", "both"])
    parser.add_argument("--out-dir", default=str(HERE / "outputs"))
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument("--workers", type=int, default=2,
                        help="Questions en parallèle. Passer à 1 en cas de rate limits répétés.")
    parser.add_argument("--no-judge", action="store_true", help="Désactive le juge LLM (et l'answer relevancy)")
    parser.add_argument("--no-answer-relevancy", action="store_true",
                        help="Désactive l'answer relevancy (1 appel Mistral Small + embeddings par réponse)")
    parser.add_argument("--time-budget", type=int, default=600,
                        help="Arrêt propre au-delà de N secondes, avec résultats partiels")
    parser.add_argument("--page-tolerance", type=int, default=1,
                        help="Écart de pagination toléré entre PDF annoté et OCR")
    parser.add_argument("--per-doc", action="store_true",
                        help="Un index par document au lieu d'un index combiné (plus facile)")
    parser.add_argument("--verbose", action="store_true", help="N'étouffe pas les logs du pipeline")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="Autorise un run partiel à écraser les sorties versionnées")
    args = parser.parse_args()

    partial = []
    if args.max_items:
        partial.append(f"--max-items {args.max_items}")
    if args.no_judge:
        partial.append("--no-judge")
    if args.docs:
        partial.append(f"--docs {args.docs}")
    if os.path.abspath(args.dataset) != os.path.abspath(HERE / "dataset.jsonl"):
        partial.append(f"--dataset {os.path.basename(args.dataset)}")  # autre protocole que celui publié
    guard_partial_overwrite(args.out_dir, str(HERE / "outputs"), partial, args.force_overwrite)

    dataset = load_dataset(args.dataset)
    n_protocol = len(dataset)  # taille du jeu complet, avant tout filtre
    if args.docs:
        wanted = {d.strip() for d in args.docs.split(",") if d.strip()}
        dataset = [ex for ex in dataset if ex.get("doc_name") in wanted]
    if args.max_items:
        dataset = dataset[: args.max_items]
    if not dataset:
        raise SystemExit("Dataset vide. Lancer d'abord prepare.py")

    docs_in_dataset = sorted({ex["doc_name"] for ex in dataset})
    k_values = [int(v) for v in args.k_values.split(",") if v.strip()]
    modes = ["baseline", "agentic"] if args.mode == "both" else [args.mode]

    _log(f"{len(dataset)} questions sur {len(docs_in_dataset)} documents: {', '.join(docs_in_dataset)}")
    _log(f"Modes: {', '.join(modes)} | workers: {args.workers} | juge: {not args.no_judge} | "
         f"answer relevancy: {not (args.no_judge or args.no_answer_relevancy)}")

    start = time.time()

    # --- Index -------------------------------------------------------------
    # Le jeu de documents indexé doit correspondre exactement au répertoire Chroma,
    # sinon BM25 et la recherche vectorielle porteraient sur des corpus différents.
    if args.per_doc:
        retrievers = {}
        for doc_name in docs_in_dataset:
            chunks = load_cached_chunks([doc_name])
            _log(f"Index {doc_name}: {len(chunks)} chunks")
            retrievers[doc_name] = build_retriever_from_chunks(
                chunks, persist_directory=str(store_dir_for([doc_name]))
            )
        get_retriever = lambda ex: retrievers[ex["doc_name"]]
    else:
        chunks = load_cached_chunks(docs_in_dataset)
        _log(f"Index combiné: {len(chunks)} chunks (recherche sur les {len(docs_in_dataset)} documents)")
        retriever = build_retriever_from_chunks(
            chunks, persist_directory=str(store_dir_for(docs_in_dataset))
        )
        get_retriever = lambda ex: retriever

    # Pages OCR entières, pour les outils grep / read_page de l'agent de recherche.
    page_store = PageStore(load_cached_pages(docs_in_dataset))
    _log(f"PageStore: {sum(page_store.page_count(d) for d in page_store.documents())} pages "
         f"sur {len(page_store.documents())} documents")

    _log(f"Index prêt en {time.time() - start:.0f}s")

    # --- Agents (instanciés une fois, nœuds sans état -> réutilisables entre threads) ---
    workflow = AgentWorkflow() if "agentic" in modes else None
    researcher = ResearchAgent() if "baseline" in modes else None
    judge = None if args.no_judge else FinanceBenchJudge()
    relevancy = None if (args.no_judge or args.no_answer_relevancy) else AnswerRelevancy()

    # Le pipeline imprime énormément de DEBUG : illisible avec plusieurs workers.
    if not args.verbose:
        sys.stdout = open(os.devnull, "w")

    per_example, errors = [], []
    truncated = False
    # Tokens facturés, cumulés par modèle sur tous les appels LLM du run (génération,
    # sous-agents, juge). Le callback vit dans un contextvar : chaque tâche du pool reçoit
    # une copie du contexte, sinon les threads ne le verraient pas.
    from langchain_core.callbacks import get_usage_metadata_callback
    from backend.retriever.builder import RERANK_USAGE
    usage_cb = get_usage_metadata_callback()
    usage = usage_cb.__enter__()
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    contextvars.copy_context().run,
                    evaluate_example, ex, get_retriever(ex), modes,
                    workflow, researcher, judge, k_values, args.page_tolerance, page_store, relevancy,
                ): ex
                for ex in dataset
            }
            done = 0
            for future in as_completed(futures):
                example = futures[future]
                done += 1
                try:
                    rows = future.result()
                    per_example.extend(rows)
                    for r in rows:
                        if r.get("failed"):
                            errors.append({
                                "id": r["id"], "mode": r["mode"],
                                "rate_limited": r.get("rate_limited"), "error": r.get("error"),
                            })
                    verdicts = "/".join(
                        ("ÉCHEC" if r.get("failed") else r.get("verdict", "ok")) for r in rows
                    )
                    _log(f"[{done}/{len(dataset)}] {example['id']} ({example['doc_name']}) -> {verdicts}")
                except Exception as e:
                    errors.append({"id": example.get("id"), "mode": "*",
                                   "type": type(e).__name__, "error": root_cause(e),
                                   "rate_limited": is_rate_limit(e)})
                    _log(f"[{done}/{len(dataset)}] {example['id']} ÉCHEC: {root_cause(e)[:160]}")

                if args.time_budget and time.time() - start > args.time_budget:
                    truncated = True
                    _log(f"⚠️  Budget de {args.time_budget}s dépassé, arrêt avec résultats partiels")
                    for pending in futures:
                        pending.cancel()
                    break
    finally:
        usage_cb.__exit__(None, None, None)
        if not args.verbose:
            sys.stdout.close()
            sys.stdout = _REAL_STDOUT

    # --- Synthèse ----------------------------------------------------------
    grouped = defaultdict(list)
    for row in per_example:
        grouped[row["mode"]].append(row)

    summary = {
        "dataset": os.path.basename(args.dataset),
        "documents": docs_in_dataset,
        "index": "per-doc" if args.per_doc else "combined",
        "n_questions": len(dataset),
        "elapsed_sec": round(time.time() - start, 2),
        "truncated": truncated,
        "errors": len(errors),
        "cost": compute_cost(usage.usage_metadata, dict(RERANK_USAGE)),
    }
    for mode in modes:
        summary[mode] = aggregate(grouped[mode], k_values)
        if grouped[mode]:
            summary[mode]["by_question_type"] = breakdown(grouped[mode], "question_type", k_values)
            summary[mode]["by_reasoning"] = breakdown(grouped[mode], "question_reasoning", k_values)

    if len(modes) == 2 and summary["baseline"] and summary["agentic"]:
        b_fb = summary["baseline"].get("financebench") or {}
        a_fb = summary["agentic"].get("financebench") or {}
        summary["delta"] = {
            "accuracy": round(a_fb.get("accuracy", 0) - b_fb.get("accuracy", 0), 4),
            "hallucination_rate": round(
                a_fb.get("hallucination_rate", 0) - b_fb.get("hallucination_rate", 0), 4),
            "mean_f1": round(summary["agentic"]["mean_f1"] - summary["baseline"]["mean_f1"], 4),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    for name, payload in [
        ("financebench_summary.json", summary),
        ("financebench_results.json", per_example),
        ("financebench_errors.json", errors),
    ]:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print_report(summary, modes, k_values, n_protocol=n_protocol)
    _log(f"Résultats écrits dans {args.out_dir}/")

    log_to_langsmith(
        name="run_financebench_eval",
        summary=summary,
        inputs={"dataset": args.dataset, "mode": args.mode, "documents": docs_in_dataset},
    )


if __name__ == "__main__":
    main()
