"""
Votes 👍/👎 sur les réponses, envoyés comme feedback LangSmith sur la trace de la réponse.

Chaque question est tracée sous un run racine dont l'identifiant est fixé par le serveur
(views.process_question). Le client ne reçoit pas cet identifiant en clair mais un jeton signé
qui le lie à son tenant : sans ce lien, connaître l'identifiant d'une trace suffirait à voter
sur la réponse d'un autre espace.

Les votes servent ensuite à evaluation/feedback/ : collect.py range les traces votées dans un
dataset LangSmith (avec la cause d'échec proposée pour les 👎), replay.py les rejoue.
"""
import uuid
from typing import Optional

from django.core import signing

from .config.settings import settings

TRACE_PROJECT = "agentic-search"
FEEDBACK_KEY = "user_score"  # 1 = satisfait, 0 = pas satisfait
COMMENT_MAX_CHARS = 2000

_SALT = "backend.feedback"
_TOKEN_MAX_AGE = 30 * 24 * 3600  # un vote n'a plus de sens un mois après la réponse


class InvalidFeedback(ValueError):
    pass


def enabled() -> bool:
    """Le vote n'existe que si les réponses sont tracées : sans trace, rien à quoi le rattacher."""
    return bool(settings.LANGSMITH_API_KEY)


def issue_token(run_id: uuid.UUID, tenant_id: str) -> str:
    return signing.dumps({"r": str(run_id), "t": tenant_id}, salt=_SALT, compress=True)


def read_token(token: str, tenant_id: str) -> uuid.UUID:
    try:
        data = signing.loads(token, salt=_SALT, max_age=_TOKEN_MAX_AGE)
    except signing.SignatureExpired as e:
        raise InvalidFeedback("Réponse trop ancienne pour être évaluée.") from e
    except signing.BadSignature as e:
        raise InvalidFeedback("Jeton de vote invalide.") from e
    if data.get("t") != tenant_id:
        raise InvalidFeedback("Jeton de vote invalide.")
    return uuid.UUID(data["r"])


def feedback_id(run_id: uuid.UUID) -> uuid.UUID:
    """Identifiant déterministe : un même vote renvoyé deux fois ne compte qu'une fois."""
    return uuid.uuid5(run_id, FEEDBACK_KEY)


def send(run_id: uuid.UUID, score: int, comment: Optional[str] = None, client=None):
    """
    `trace_id` et le client du traceur font passer le feedback par la même file que le run,
    derrière lui : le vote peut arriver avant que la trace ait fini d'être envoyée.
    """
    if client is None:
        from langsmith.run_trees import get_cached_client
        client = get_cached_client()
    client.create_feedback(
        run_id,
        key=FEEDBACK_KEY,
        score=score,
        comment=comment or None,
        trace_id=run_id,
        feedback_id=feedback_id(run_id),
    )
