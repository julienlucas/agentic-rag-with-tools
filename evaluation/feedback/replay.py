"""
Rejoue les retours utilisateurs comme tests de non-régression, à lancer à la main avant de
merger un changement de prompt, d'outil, de retrieval ou de modèle (comme regression.py de
FinanceBench). Pas en CI : il appelle les vraies API.

Seuls les exemples du dataset « Retours utilisateurs » qui ont une réponse de référence
(outputs.answer) sont rejoués, sur les documents de l'espace où la question a été posée :
même tenant, mêmes documents (vérifiés par hash). Un document supprimé depuis rend l'exemple
non rejouable, il est sauté.

- 👍 : la réponse approuvée est la référence. Ne plus la retrouver est une RÉGRESSION.
- 👎 avec une référence écrite à la main : mesure les corrections (CORRIGÉ / TOUJOURS FAUX).
  Un 👎 encore faux ne fait pas échouer le garde-fou : c'est un défaut connu, pas une régression.

    uv run python evaluation/feedback/replay.py
    uv run python evaluation/feedback/replay.py --push --name "prompt v3"   # expérience LangSmith

Code de sortie : 0 si aucune régression, 1 sinon, 2 si rien n'a pu être jugé.
"""
import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import settings  # noqa: E402  (charge aussi le .env)
from backend.feedback_dataset import DATASET, cited_passages  # noqa: E402

HERE = Path(__file__).resolve().parent
LAST_RUN = HERE / "last_run.json"  # non versionné : contient les questions des utilisateurs

# Le protocole du juge FinanceBench (verdict ternaire + faithfulness), sans le contexte financier :
# les documents des utilisateurs sont de toute nature.
JUDGE_PROMPT = """Tu es un évaluateur rigoureux. Tu compares la réponse d'un système RAG à une réponse de référence validée par un utilisateur.

**Question:** {question}

**Réponse de référence:** {expected_answer}

**Remarque sur la référence:** {justification}

**Réponse générée par le système:** {generated_answer}

**Extraits du document fournis au système:**
{context}

Rends DEUX jugements.

1) VERDICT — classe la réponse générée:
- "CORRECT": elle donne la même information que la référence. Tolère les écarts de formulation, d'unité, d'arrondi et de mise en forme. Une réponse plus ou moins détaillée reste CORRECT si elle contient l'information essentielle de la référence et ne la contredit pas.
- "INCORRECT": elle contredit la référence, donne un chiffre faux, omet l'information essentielle, ou répond à côté.
- "REFUSAL": elle déclare ne pas pouvoir répondre ou que l'information n'est pas dans le document.

Si la référence dit elle-même que l'information n'est pas dans le document, une réponse générée qui le constate est CORRECT.

2) FAITHFULNESS (1-5) — tout ce qu'affirme la réponse est-il appuyé par les extraits fournis ?
- 5: intégralement appuyé | 4: inférences mineures raisonnables | 3: quelques affirmations non appuyées
- 2: plusieurs affirmations inventées | 1: hallucinations majeures
Un refus honnête vaut 5.

Réponds EXACTEMENT dans ce format:
VERDICT: [CORRECT|INCORRECT|REFUSAL]
FAITHFULNESS: [1-5]
RAISON: [1-2 phrases]"""


def _log(msg: str):
    print(f"[replay] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Fonctions pures (testées dans tests/test_feedback.py)
# ---------------------------------------------------------------------------

def replayable(examples: List[Dict]) -> List[Dict]:
    """Les exemples qui ont une référence non vide ; un 👎 sans référence attend qu'on l'écrive."""
    return [ex for ex in examples if ((ex.get("outputs") or {}).get("answer") or "").strip()]


def outcome(vote: str, verdict: str) -> str:
    if verdict == "ERROR":
        return "ERREUR"
    ok = verdict == "CORRECT"
    if vote == "up":
        return "OK" if ok else "RÉGRESSION"
    return "CORRIGÉ" if ok else "TOUJOURS FAUX"


def summarize(rows: List[Dict]) -> Dict:
    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    judged = sum(n for k, n in counts.items() if k not in ("ERREUR", "SAUTÉ"))
    return {
        "counts": counts,
        "judged": judged,
        "regressions": [r for r in rows if r["outcome"] == "RÉGRESSION"],
        "exit_code": 2 if judged == 0 else (1 if counts.get("RÉGRESSION") else 0),
    }


# ---------------------------------------------------------------------------
# Rejeu
# ---------------------------------------------------------------------------

def _context(answer: str, citations: List[Dict]) -> str:
    return cited_passages(answer, citations, max_chars=6000)


def replay_example(ex: Dict, workflow, judge, store) -> Dict:
    from backend.retriever.builder import RetrieverBuilder
    from backend.retriever.page_store import PageStore

    meta = ex.get("metadata") or {}
    question = ex["inputs"]["question"]
    row = {"example_id": ex["id"], "run_id": meta.get("run_id"), "vote": meta.get("vote", "up"),
           "question": question, "expected": ex["outputs"]["answer"]}

    try:
        tenant = store.for_tenant(meta.get("tenant_id") or "")
    except ValueError:
        return {**row, "outcome": "SAUTÉ", "reason": "tenant inconnu"}
    missing = [h for h in meta.get("documents") or [] if not tenant.has_document(h)]
    if missing or not meta.get("documents"):
        return {**row, "outcome": "SAUTÉ", "reason": f"{len(missing)} document(s) supprimé(s) depuis le vote"}

    started = time.time()
    pages = tenant.pages()
    result = workflow.full_pipeline(
        question=question,
        retriever=RetrieverBuilder().build_hybrid_retriever(tenant),
        page_store=PageStore(pages) if pages else None,
    )
    answer = result["draft_answer"]
    verdict = judge.evaluate(
        question=question,
        expected_answer=row["expected"],
        generated_answer=answer,
        context=_context(answer, result.get("citations") or []),
        justification=meta.get("comment") or "",
    )
    return {**row, "answer": answer, "verdict": verdict.verdict, "faithfulness": verdict.faithfulness,
            "reason": verdict.reason, "seconds": round(time.time() - started, 1),
            "outcome": outcome(row["vote"], verdict.verdict)}


def push_experiment(client, dataset, rows: List[Dict], name: str) -> str:
    """Une expérience LangSmith rattachée au dataset : les rejeux successifs se comparent côte à côte."""
    project = client.create_project(
        f"{name} · retours · {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        reference_dataset_id=dataset.id,
        metadata={"reasoning_model": settings.REASONING_MODEL_ID, "small_model": settings.MODEL_SMALL_ID},
    )
    end = datetime.now(timezone.utc)
    runs = []
    for row in rows:
        if "verdict" not in row:
            continue
        run_id = uuid.uuid4()
        client.create_run(
            id=run_id, name="replay", run_type="chain",
            inputs={"question": row["question"]}, outputs={"answer": row["answer"]},
            reference_example_id=row["example_id"], project_name=project.name,
            start_time=end - timedelta(seconds=row["seconds"]), end_time=end,
            extra={"metadata": {"vote": row["vote"], "outcome": row["outcome"]}},
        )
        runs.append((run_id, row))
    client.flush()  # les runs doivent exister avant leurs feedbacks
    for run_id, row in runs:
        if row["verdict"] != "ERROR":
            client.create_feedback(run_id, key="correct", score=1.0 if row["verdict"] == "CORRECT" else 0.0,
                                   comment=f"{row['verdict']} — {row['reason']}")
            client.create_feedback(run_id, key="judge_faithfulness", score=row["faithfulness"])
    client.update_project(project.id, end_time=datetime.now(timezone.utc))
    return project.name


def main():
    parser = argparse.ArgumentParser(description="Rejoue les retours utilisateurs (non-régression)")
    parser.add_argument("--limit", type=int, default=None, help="Au plus N exemples (les plus récents)")
    parser.add_argument("--push", action="store_true", help="Envoyer le rejeu comme expérience LangSmith")
    parser.add_argument("--name", default=None, help="Nom de l'expérience (défaut : le modèle de réponse)")
    args = parser.parse_args()

    if not settings.LANGSMITH_API_KEY:
        _log("LANGSMITH_API_KEY non défini : dataset inaccessible")
        sys.exit(2)
    from langsmith import Client
    from backend.agents.workflow import AgentWorkflow
    from backend.vectorstore import get_store
    from evaluation.llm_judge import FinanceBenchJudge

    client = Client()
    if not client.has_dataset(dataset_name=DATASET):
        _log(f"dataset « {DATASET} » absent : lancer d'abord collect.py")
        sys.exit(2)
    dataset = client.read_dataset(dataset_name=DATASET)
    examples = [
        {"id": ex.id, "inputs": ex.inputs, "outputs": ex.outputs, "metadata": ex.metadata}
        for ex in client.list_examples(dataset_id=dataset.id)
    ]
    todo = replayable(examples)
    todo.sort(key=lambda ex: (ex["metadata"] or {}).get("voted_at", ""), reverse=True)
    todo = todo[: args.limit] if args.limit else todo
    _log(f"{len(todo)} exemple(s) avec référence sur {len(examples)} dans « {DATASET} »")

    workflow, judge, store = AgentWorkflow(), FinanceBenchJudge(prompt=JUDGE_PROMPT), get_store()
    rows = []
    for i, ex in enumerate(todo, 1):
        row = replay_example(ex, workflow, judge, store)
        rows.append(row)
        _log(f"{i}/{len(todo)} {row['outcome']:<14} {row['question'][:70]!r}"
             + (f" — {row['reason']}" if row["outcome"] in ("RÉGRESSION", "SAUTÉ", "ERREUR") else ""))

    summary = summarize(rows)
    LAST_RUN.write_text(json.dumps({"summary": {k: v for k, v in summary.items() if k != "regressions"},
                                    "rows": rows}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _log(" · ".join(f"{k} {n}" for k, n in sorted(summary["counts"].items())) or "rien à rejouer")
    if args.push and summary["judged"]:
        _log(f"expérience LangSmith « {push_experiment(client, dataset, rows, args.name or settings.REASONING_MODEL_ID)} »")
    sys.exit(summary["exit_code"])


if __name__ == "__main__":
    main()
