"""
Résilience aux rate limits — partagé entre le backend (workflow, agents) et l'évaluation.
"""
import time
from typing import Optional


def is_rate_limit(exc: Exception) -> bool:
    """Reconnaît un 429, quelle que soit la forme sous laquelle l'erreur remonte."""
    cur = exc
    while cur is not None:
        if getattr(cur, "status_code", None) == 429:
            return True
        text = f"{type(cur).__name__}: {cur}".lower()
        if "429" in text or "rate limit" in text or "rate_limited" in text \
           or "too many requests" in text or "capacity exceeded" in text \
           or "service_tier_capacity_exceeded" in text \
           or "throttlingexception" in text or "too many tokens" in text:  # Bedrock
            return True
        cur = cur.__cause__
    return False


def retry_after(exc: Exception) -> Optional[float]:
    """Lit l'en-tête Retry-After si l'API en fournit un."""
    cur = exc
    while cur is not None:
        headers = getattr(getattr(cur, "raw_response", None), "headers", None)
        if headers:
            for key in ("retry-after", "Retry-After", "x-ratelimit-reset"):
                value = headers.get(key)
                if value:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
        cur = cur.__cause__
    return None


def root_cause(exc: Exception) -> str:
    """Déroule la chaîne des __cause__ (les wrappers du projet masquent l'erreur d'origine)."""
    parts, cur, seen = [], exc, 0
    while cur is not None and seen < 5:
        parts.append(f"{type(cur).__name__}: {cur}")
        cur, seen = cur.__cause__, seen + 1
    return " <- ".join(parts)


def call_with_backoff(fn, what: str, attempts: int = 6, base_delay: float = 10.0,
                      log=print):
    """Rejoue un appel API en cas de rate limit, avec backoff exponentiel."""
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if not is_rate_limit(e) or attempt == attempts:
                raise
            wait = min(retry_after(e) or delay, 120.0)
            log(f"rate limit sur {what} (tentative {attempt}/{attempts}), attente {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, 120.0)
