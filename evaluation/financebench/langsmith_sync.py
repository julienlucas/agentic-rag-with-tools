"""
Envoi de l'évaluation FinanceBench sur LangSmith.

- Le dataset (questions, réponses de référence, document, type de question) devient un
  dataset LangSmith, synchronisé par identifiant FinanceBench : relancer n'ajoute que les
  questions absentes.
- Chaque run devient une expérience par mode (baseline / agentic) rattachée à ce dataset :
  une ligne par question, avec la réponse générée et ses scores en feedback (verdict du juge,
  F1, faithfulness, recall@10...). Les expériences successives se comparent côte à côte
  dans l'onglet « Experiments » du dataset.

Appelé à la fin de run_financebench_eval.py. Utilisable aussi après coup, sur un dossier de
sorties existant (aucun appel de modèle, rien n'est recalculé) :

    uv run python evaluation/financebench/langsmith_sync.py \\
        --out-dir evaluation/financebench/outputs_sonnet5_5 --name "Claude Sonnet 5.5"
"""
import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import settings  # noqa: E402  (charge aussi le .env)
from evaluation.utils import load_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent

# Scores numériques d'une ligne de résultats, envoyés comme feedback LangSmith.
NUMERIC_FEEDBACK = [
    "answer_f1", "judge_faithfulness", "answer_relevancy", "tool_calls",
    "recall@5", "recall@10", "mrr@10", "ndcg@10", "page_hit@5", "page_hit@10", "page_recall@10",
]


def get_client():
    if not os.getenv("LANGSMITH_API_KEY"):
        print("⚠️  LANGSMITH_API_KEY non défini : rien n'est envoyé à LangSmith")
        return None
    from langsmith import Client
    return Client()


def dataset_name_for(dataset_path: str) -> str:
    stem = Path(dataset_path).stem
    return "FinanceBench" if stem == "dataset" else f"FinanceBench ({stem})"


def sync_dataset(client, dataset_path: str):
    """Crée le dataset s'il n'existe pas et ajoute les questions manquantes. Retourne (dataset, {id: example_id})."""
    rows = load_dataset(dataset_path)
    name = dataset_name_for(dataset_path)
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(
            name, description="FinanceBench (Patronus AI) : questions sur des rapports 10-K, réponses de référence.",
        )

    def by_id():
        return {
            (ex.metadata or {}).get("financebench_id"): ex.id
            for ex in client.list_examples(dataset_id=dataset.id)
        }

    known = by_id()
    missing = [r for r in rows if r["id"] not in known]
    if missing:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"question": r["question"], "doc_name": r["doc_name"]} for r in missing],
            outputs=[{"answer": r["expected_answer"], "justification": r.get("justification", "")} for r in missing],
            metadata=[{
                "financebench_id": r["id"], "company": r.get("company"), "doc_name": r["doc_name"],
                "question_type": r.get("question_type"), "question_reasoning": r.get("question_reasoning"),
            } for r in missing],
        )
        known = by_id()
        print(f"✓ Dataset LangSmith « {name} » : {len(missing)} question(s) ajoutée(s), {len(known)} au total")
    return dataset, known


def _feedback(row: Dict) -> List[tuple]:
    out = []
    verdict = row.get("verdict")
    if verdict and verdict != "ERROR":
        out.append(("correct", 1.0 if verdict == "CORRECT" else 0.0, f"{verdict} — {row.get('judge_reason', '')}"))
    for key in NUMERIC_FEEDBACK:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append((key, float(value), None))
    if isinstance(row.get("evidence_seen"), bool):
        out.append(("evidence_seen", 1.0 if row["evidence_seen"] else 0.0, None))
    return out


def log_experiments(client, dataset_path: str, summary: Dict, results: List[Dict],
                    name: Optional[str] = None) -> List[str]:
    """Une expérience LangSmith par mode, une ligne par question. Retourne les noms des expériences."""
    dataset, example_ids = sync_dataset(client, dataset_path)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    name = name or settings.REASONING_MODEL_ID
    experiments = []

    for mode in sorted({r["mode"] for r in results}):
        rows = [r for r in results if r["mode"] == mode and r.get("id") in example_ids]
        fb = (summary.get(mode) or {}).get("financebench") or {}
        project = client.create_project(
            f"{name} · {mode} · {stamp}",
            reference_dataset_id=dataset.id,
            metadata={
                "mode": mode,
                "reasoning_model": name,
                "small_model": settings.MODEL_SMALL_ID,
                "embeddings": settings.EMBEDDING_MODEL_ID,
                "reranker": settings.RERANK_MODEL,
                "index": summary.get("index"),
                "documents": summary.get("documents"),
                "accuracy": fb.get("accuracy"),
                "cost_usd": (summary.get("cost") or {}).get("total_usd"),
            },
        )
        run_ids = []
        end = datetime.now(timezone.utc)
        for row in rows:
            run_id = uuid.uuid4()
            duration = float(row.get("retrieval_sec") or 0) + float(row.get("generation_sec") or 0)
            client.create_run(
                id=run_id,
                name="financebench",
                run_type="chain",
                inputs={"question": row["question"], "doc_name": row["doc_name"]},
                outputs={"answer": row.get("answer", "")},
                reference_example_id=example_ids[row["id"]],
                project_name=project.name,
                start_time=end - timedelta(seconds=duration),
                end_time=end,
                extra={"metadata": {
                    "mode": mode, "verdict": row.get("verdict"), "question_type": row.get("question_type"),
                    "pages_read": row.get("pages_read"), "tool_queries": row.get("corrective_queries"),
                    "gold_rank": row.get("gold_rank"),
                }},
            )
            run_ids.append((run_id, row))
        client.flush()  # les runs doivent exister avant leurs feedbacks
        for run_id, row in run_ids:
            for key, score, comment in _feedback(row):
                client.create_feedback(run_id, key=key, score=score, comment=comment)
        client.update_project(project.id, end_time=datetime.now(timezone.utc))
        experiments.append(project.name)
        acc = fb.get("accuracy")
        print(f"✓ Expérience LangSmith « {project.name} » : {len(rows)} questions"
              + (f", exactitude {acc:.1%}" if isinstance(acc, (int, float)) else ""))
    return experiments


def push_eval(dataset_path: str, summary: Dict, results: List[Dict], name: Optional[str] = None) -> List[str]:
    """Point d'entrée du runner : n'échoue jamais (un incident LangSmith ne doit pas faire perdre un run)."""
    client = get_client()
    if client is None:
        return []
    try:
        return log_experiments(client, dataset_path, summary, results, name)
    except Exception as e:
        print(f"⚠️  Envoi LangSmith impossible : {type(e).__name__}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Envoie un run FinanceBench existant sur LangSmith")
    parser.add_argument("--out-dir", required=True, help="Dossier contenant financebench_summary.json et _results.json")
    parser.add_argument("--name", default=None, help="Nom de l'expérience (ex. le modèle évalué)")
    parser.add_argument("--dataset", default=None, help="Jeu de questions (défaut : celui du run)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    summary = json.loads((out / "financebench_summary.json").read_text(encoding="utf-8"))
    results = json.loads((out / "financebench_results.json").read_text(encoding="utf-8"))
    dataset_path = args.dataset or str(HERE / summary.get("dataset", "dataset.jsonl"))
    if not push_eval(dataset_path, summary, results, args.name):
        sys.exit(1)


if __name__ == "__main__":
    main()
