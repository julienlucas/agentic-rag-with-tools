"""
Range les votes des utilisateurs dans le dataset LangSmith « Retours utilisateurs ».

Chaque vote (feedback `user_score` posé par /api/feedback sur la trace d'une réponse) devient
un exemple du dataset, une seule fois par trace :

- 👍 : la réponse approuvée devient la réponse de référence (outputs.answer). L'exemple est
  aussitôt un test de non-régression : replay.py vérifie qu'on répond toujours aussi bien.
- 👎 : pas de référence (on sait seulement que la réponse donnée ne convenait pas). La réponse
  rejetée et le commentaire vont en métadonnées, avec une cause d'échec proposée par un LLM
  (aussi posée en feedback `failure_category` sur la trace, filtrable dans LangSmith). Pour
  en faire un test, écrire la bonne réponse dans outputs.answer depuis l'interface LangSmith,
  et corriger la cause si le classement s'est trompé.

Le classement utilise le juge de l'évaluation (Mistral sur Bedrock), pas Claude : le modèle
qui a répondu ne juge pas ses propres échecs.

    uv run python evaluation/feedback/collect.py
    uv run python evaluation/feedback/collect.py --no-classify   # sans appel LLM
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import settings  # noqa: E402  (charge aussi le .env)
from backend.feedback import FEEDBACK_KEY, TRACE_PROJECT  # noqa: E402

DATASET = "Retours utilisateurs"
CATEGORY_KEY = "failure_category"

# Causes d'échec : ce qu'on corrige n'est pas au même endroit selon la cause.
CATEGORIES = {
    "mauvais_document": "la réponse s'appuie sur un autre document que celui que vise la question",
    "passage_manquant": "le passage qui contient la réponse n'a pas été trouvé (réponse fausse ou incomplète faute de preuve)",
    "erreur_lecture": "le bon passage est cité mais mal lu : mauvaise ligne, mauvaise période, mauvaise unité",
    "erreur_calcul": "les chiffres sont les bons mais le calcul (ratio, variation, somme) est faux",
    "refus_a_tort": "le système dit que l'information n'est pas disponible alors qu'elle l'est",
    "hors_corpus": "la réponse n'est pas dans les documents : le refus était justifié, ou la question est hors sujet",
    "forme": "le fond est juste mais la réponse est trop longue, confuse ou ne répond pas directement",
    "autre": "aucune des causes ci-dessus",
}

CLASSIFY_PROMPT = """Un utilisateur a jugé insatisfaisante la réponse d'un système de questions-réponses sur documents (RAG). Identifie la cause la plus probable.

**Question:** {question}

**Réponse donnée:** {answer}

**Commentaire de l'utilisateur:** {comment}

**Passages cités par la réponse:**
{passages}

**Rapport du pipeline (recherches et outils utilisés):**
{report}

Causes possibles:
{categories}

Réponds EXACTEMENT dans ce format:
CAUSE: [une des causes ci-dessus, en toutes lettres]
RAISON: [1 phrase]"""


def _log(msg: str):
    print(f"[feedback] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Fonctions pures (testées dans tests/test_feedback.py)
# ---------------------------------------------------------------------------

def parse_classification(text: str) -> Tuple[str, str]:
    m = re.search(r"CAUSE\**\s*:[\s*\[]*([a-z_]+)", text or "", re.IGNORECASE)
    category = m.group(1).lower() if m else "autre"
    if category not in CATEGORIES:
        category = "autre"
    m = re.search(r"RAISON\**\s*:\s*(.+?)(?:\n\s*\n|$)", text or "", re.IGNORECASE | re.DOTALL)
    reason = " ".join(m.group(1).split()) if m else ""
    return category, reason


def cited_passages(answer: str, citations: List[Dict], max_chars: int = 4000) -> str:
    """Les passages réellement cités [n] dans la réponse, comme le frontend les affiche."""
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer or "")}
    lines = [f"[{c.get('n')}] {c.get('locator', '')} — {c.get('excerpt', '')}"
             for c in citations or [] if c.get("n") in used]
    return "\n".join(lines)[:max_chars] or "(aucun)"


def example_from_vote(run_id: str, question: str, outputs: Dict, run_metadata: Dict, score: float,
                      comment: Optional[str], voted_at: str,
                      category: Optional[str] = None, reason: str = "") -> Dict:
    """L'exemple du dataset pour un vote : référence si 👍, réponse rejetée en métadonnée si 👎."""
    answer = (outputs or {}).get("draft_answer", "")
    up = score >= 1
    metadata = {
        "run_id": run_id,
        "vote": "up" if up else "down",
        "comment": comment or "",
        "voted_at": voted_at,
        "tenant_id": run_metadata.get("tenant_id"),
        "documents": run_metadata.get("documents") or [],
        "document_names": run_metadata.get("document_names") or [],
    }
    if not up:
        metadata.update({"rejected_answer": answer, CATEGORY_KEY: category or "", "failure_reason": reason})
    return {
        "inputs": {"question": question},
        "outputs": {"answer": answer} if up else None,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------

def get_dataset(client):
    if client.has_dataset(dataset_name=DATASET):
        return client.read_dataset(dataset_name=DATASET)
    return client.create_dataset(
        DATASET,
        description=(
            f"Votes des utilisateurs sur les réponses (projet {TRACE_PROJECT}). 👍 : la réponse fait "
            "référence. 👎 : réponse rejetée en métadonnées, référence à écrire dans outputs.answer."
        ),
    )


def classify(llm, question: str, outputs: Dict, comment: Optional[str]) -> Tuple[str, str]:
    answer = outputs.get("draft_answer", "")
    prompt = CLASSIFY_PROMPT.format(
        question=question,
        answer=answer[:3000],
        comment=comment or "(aucun)",
        passages=cited_passages(answer, outputs.get("citations") or []),
        report=(outputs.get("verification_report") or "(aucun)")[:2000],
        categories="\n".join(f"- {k} : {v}" for k, v in CATEGORIES.items()),
    )
    return parse_classification(llm.invoke(prompt).content)


def collect(client, llm=None) -> Dict[str, Counter]:
    dataset = get_dataset(client)
    known = {(ex.metadata or {}).get("run_id") for ex in client.list_examples(dataset_id=dataset.id)}
    added = {"up": Counter(), "down": Counter()}

    for fb in client.list_feedback(feedback_key=[FEEDBACK_KEY]):
        run_id = str(fb.run_id)
        if run_id in known or fb.score is None:
            continue
        run = client.read_run(fb.run_id)
        question = (run.inputs or {}).get("question", "")
        outputs = run.outputs or {}
        if not question or not outputs.get("draft_answer"):
            _log(f"trace {run_id} incomplète (question ou réponse absente), ignorée")
            continue

        category, reason = None, ""
        if fb.score < 1 and llm is not None:
            try:
                category, reason = classify(llm, question, outputs, fb.comment)
            except Exception as e:
                _log(f"classement impossible pour {run_id} : {type(e).__name__}: {e}")

        ex = example_from_vote(
            run_id, question, outputs, (run.extra or {}).get("metadata") or {}, fb.score, fb.comment,
            fb.created_at.isoformat() if fb.created_at else "", category, reason,
        )
        client.create_example(dataset_id=dataset.id, source_run_id=run.id, **ex)
        if category:
            client.create_feedback(run.id, key=CATEGORY_KEY, value=category, comment=reason)
        known.add(run_id)
        vote = ex["metadata"]["vote"]
        added[vote][(category or "non classé") if vote == "down" else "référence"] += 1
        _log(f"{'👍' if vote == 'up' else '👎'} {question[:70]!r}" + (f" → {category}" if category else ""))
    return added


def main():
    parser = argparse.ArgumentParser(description="Range les votes utilisateurs dans un dataset LangSmith")
    parser.add_argument("--no-classify", action="store_true", help="Ne pas proposer de cause d'échec (aucun appel LLM)")
    args = parser.parse_args()

    if not settings.LANGSMITH_API_KEY:
        _log("LANGSMITH_API_KEY non défini : aucun vote à lire")
        sys.exit(1)
    from langsmith import Client
    llm = None
    if not args.no_classify:
        from backend.llm.bedrock import chat_model
        llm = chat_model(settings.EVAL_JUDGE_MODEL_ID, max_tokens=200, region=settings.EVAL_JUDGE_REGION)

    added = collect(Client(), llm)
    up, down = sum(added["up"].values()), sum(added["down"].values())
    _log(f"{up} 👍 et {down} 👎 ajoutés au dataset « {DATASET} »")
    for category, n in added["down"].most_common():
        _log(f"  {category:<18} {n}")


if __name__ == "__main__":
    main()
