"""
Range un vote dans le dataset LangSmith « Retours utilisateurs », une seule fois par trace :

- 👍 : la réponse approuvée devient la réponse de référence (outputs.answer). L'exemple est
  aussitôt un test de non-régression (evaluation/feedback/replay.py).
- 👎 : pas de référence (on sait seulement que la réponse donnée ne convenait pas). La réponse
  rejetée et le commentaire vont en métadonnées, avec une cause d'échec proposée par un LLM
  (aussi posée en feedback `failure_category` sur la trace, filtrable dans LangSmith). Pour
  en faire un test, écrire la bonne réponse dans outputs.answer depuis l'interface LangSmith,
  et corriger la cause si le classement s'est trompé.

Appelé en arrière-plan après chaque vote (collect_in_background), et en lot par
evaluation/feedback/collect.py, qui rattrape les votes que l'arrière-plan a manqués
(redémarrage du serveur, trace trop lente à arriver, LangSmith indisponible).

Le classement utilise le juge de l'évaluation (Mistral sur Bedrock), pas Claude : le modèle
qui a répondu ne juge pas ses propres échecs.
"""
import re
import threading
import time
import uuid
from typing import Dict, List, Optional, Set, Tuple

from .config.settings import settings
from .feedback import TRACE_PROJECT
from .utils.logging import logger

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

# Attentes avant chaque tentative en arrière-plan (~2 min 15 au total) : la trace part dans la
# file du traceur et peut arriver après le vote, surtout pour une réponse longue.
RETRY_DELAYS = (5, 10, 20, 40, 60)


class TraceNotReady(Exception):
    """La trace votée n'est pas (encore) dans LangSmith, ou pas terminée."""


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


_judge = None


def judge_llm():
    global _judge
    if _judge is None:
        from .llm.bedrock import chat_model
        _judge = chat_model(settings.EVAL_JUDGE_MODEL_ID, max_tokens=200, region=settings.EVAL_JUDGE_REGION)
    return _judge


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


def add_vote(client, run_id, score: float, comment: Optional[str], voted_at: str,
             llm=None, dataset=None, known: Optional[Set[str]] = None) -> Optional[Dict]:
    """
    Range un vote. Retourne l'exemple créé, ou None si la trace est déjà dans le dataset.
    `known` : run_id déjà rangés (le lot les charge une fois) ; sinon le dataset est interrogé.
    Lève TraceNotReady si la trace n'est pas encore complète dans LangSmith.
    """
    from langsmith.utils import LangSmithNotFoundError

    run_id = str(run_id)
    dataset = dataset or get_dataset(client)
    if known is not None:
        if run_id in known:
            return None
    elif next(iter(client.list_examples(dataset_id=dataset.id, metadata={"run_id": run_id}, limit=1)), None):
        return None

    try:
        run = client.read_run(run_id)
    except LangSmithNotFoundError as e:
        raise TraceNotReady(run_id) from e
    question = (run.inputs or {}).get("question", "")
    outputs = run.outputs or {}
    if run.end_time is None or not question or not outputs.get("draft_answer"):
        raise TraceNotReady(run_id)

    category, reason = None, ""
    if score < 1 and llm is not None:
        try:
            category, reason = classify(llm, question, outputs, comment)
        except Exception as e:
            logger.warning(f"Vote {run_id} : classement impossible ({type(e).__name__}: {e})")

    ex = example_from_vote(run_id, question, outputs, (run.extra or {}).get("metadata") or {},
                           score, comment, voted_at, category, reason)
    client.create_example(dataset_id=dataset.id, source_run_id=run.id, **ex)
    if category:
        client.create_feedback(run.id, key=CATEGORY_KEY, value=category, comment=reason)
    if known is not None:
        known.add(run_id)
    return ex


def _collect_with_retries(run_id: uuid.UUID, score: int, comment: Optional[str], voted_at: str,
                          delays=RETRY_DELAYS, client=None, llm=None):
    if client is None:
        from langsmith import Client
        client = Client()
    for delay in delays:
        time.sleep(delay)
        try:
            add_vote(client, run_id, score, comment, voted_at, llm=llm if llm is not None else judge_llm())
            return
        except TraceNotReady:
            continue
        except Exception as e:
            logger.warning(f"Vote {run_id} non rangé dans le dataset ({type(e).__name__}: {e}) : collect.py le rattrapera")
            return
    logger.warning(f"Vote {run_id} : trace toujours absente de LangSmith, collect.py le rattrapera")


def collect_in_background(run_id: uuid.UUID, score: int, comment: Optional[str]):
    """Range le vote sans faire attendre l'utilisateur ; un échec est rattrapé par collect.py."""
    if not settings.FEEDBACK_AUTO_COLLECT:
        return
    voted_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    threading.Thread(target=_collect_with_retries, args=(run_id, score, comment, voted_at),
                     name=f"feedback-{run_id}", daemon=True).start()
