"""
Range en lot les votes des utilisateurs dans le dataset LangSmith « Retours utilisateurs ».

Le serveur range déjà chaque vote en arrière-plan, quelques secondes après le clic
(backend/feedback_dataset.py, qui décrit aussi le contenu du dataset). Ce script rattrape ceux
qu'il a manqués : serveur redémarré entre-temps, trace arrivée trop tard, LangSmith ou Bedrock
indisponible. Le relancer ne crée pas de doublon (un exemple par trace).

    uv run python evaluation/feedback/collect.py
    uv run python evaluation/feedback/collect.py --no-classify   # sans appel LLM
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import settings  # noqa: E402  (charge aussi le .env)
from backend.feedback import FEEDBACK_KEY  # noqa: E402
from backend.feedback_dataset import DATASET, TraceNotReady, add_vote, get_dataset, judge_llm  # noqa: E402


def _log(msg: str):
    print(f"[feedback] {msg}", flush=True)


def collect(client, llm=None) -> Dict[str, Counter]:
    dataset = get_dataset(client)
    known = {(ex.metadata or {}).get("run_id") for ex in client.list_examples(dataset_id=dataset.id)}
    added = {"up": Counter(), "down": Counter()}

    for fb in client.list_feedback(feedback_key=[FEEDBACK_KEY]):
        if fb.run_id is None or fb.score is None:
            continue
        try:
            ex = add_vote(client, fb.run_id, fb.score, fb.comment,
                          fb.created_at.isoformat() if fb.created_at else "",
                          llm=llm, dataset=dataset, known=known)
        except TraceNotReady:
            _log(f"trace {fb.run_id} incomplète (question ou réponse absente), ignorée")
            continue
        if ex is None:
            continue
        meta = ex["metadata"]
        if meta["vote"] == "up":
            added["up"]["référence"] += 1
        else:
            added["down"][meta.get("failure_category") or "non classé"] += 1
        _log(f"{'👍' if meta['vote'] == 'up' else '👎'} {ex['inputs']['question'][:70]!r}"
             + (f" → {meta['failure_category']}" if meta.get("failure_category") else ""))
    return added


def main():
    parser = argparse.ArgumentParser(description="Range les votes utilisateurs dans un dataset LangSmith")
    parser.add_argument("--no-classify", action="store_true", help="Ne pas proposer de cause d'échec (aucun appel LLM)")
    args = parser.parse_args()

    if not settings.LANGSMITH_API_KEY:
        _log("LANGSMITH_API_KEY non défini : aucun vote à lire")
        sys.exit(1)
    from langsmith import Client

    added = collect(Client(), None if args.no_classify else judge_llm())
    up, down = sum(added["up"].values()), sum(added["down"].values())
    _log(f"{up} 👍 et {down} 👎 rattrapés dans le dataset « {DATASET} »")
    for category, n in added["down"].most_common():
        _log(f"  {category:<18} {n}")


if __name__ == "__main__":
    main()
