"""
Answer relevancy : la réponse traite-t-elle la question posée ?

Méthode de RAGAS (Es et al., 2023) : un LLM écrit les questions auxquelles la réponse répond,
puis on mesure la similarité cosinus moyenne entre leurs embeddings et celui de la question
d'origine. Une réponse hors sujet, ou qui répond à côté (autre exercice, autre métrique),
produit des questions éloignées ; une réponse évasive ou un refus vaut 0.

Complémentaire du verdict du juge, pas redondant : le juge dit si la réponse est JUSTE par
rapport à la référence ; la relevancy dit si elle est CENTRÉE sur la question — une réponse
correcte noyée dans trois paragraphes hors sujet a un bon verdict et une relevancy basse.
Ne compare jamais à la réponse de référence.
"""
import math
import re
from typing import List, Optional

from langchain_mistralai import ChatMistralAI

from backend.config.settings import settings
from backend.utils.resilience import is_rate_limit
from evaluation.llm_judge import FinanceBenchJudge

RELEVANCY_PROMPT = """Voici une réponse produite par un assistant d'analyse financière.

Réponse :
{answer}

1. Sur la première ligne, écris EVASIVE: OUI si la réponse ne s'engage pas (refus, « information
   non disponible », réponse vague sans contenu), sinon EVASIVE: NON.
2. Puis écris {n} questions différentes auxquelles cette réponse répond directement, une par
   ligne, dans la langue de la réponse, sans numérotation. Chaque question doit reprendre
   l'entreprise, la période et la métrique dont parle la réponse."""


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _strip_citations(text: str) -> str:
    return re.sub(r"(\[\d+\])+", "", text or "").strip()


def parse_generated(content: str, n: int):
    """(évasive, questions) depuis la sortie du LLM, tolérante aux puces et numéros."""
    evasive = False
    questions = []
    for raw in (content or "").splitlines():
        line = raw.strip().strip("*").strip()
        if not line:
            continue
        m = re.match(r"EVASIVE\s*:\s*\**\s*(OUI|NON|YES|NO)\b", line, re.IGNORECASE)
        if m:
            evasive = m.group(1).upper() in ("OUI", "YES")
            continue
        line = re.sub(r"^(\d+[.)]|[-•])\s*", "", line).strip()
        if len(line) > 8:
            questions.append(line)
    return evasive, questions[:n]


class AnswerRelevancy:
    def __init__(self, llm=None, embeddings=None, n_questions: int = 3):
        self.llm = llm
        self.embeddings = embeddings
        self.n = n_questions

    def _get_llm(self):
        if self.llm is None:
            self.llm = ChatMistralAI(
                model=settings.MODEL_SMALL_ID,
                api_key=settings.MISTRALAI_API_KEY,
                temperature=0,
                max_tokens=300,
                timeout=settings.LLM_TIMEOUT,
                max_retries=settings.LLM_MAX_RETRIES,
            )
        return self.llm

    def _get_embeddings(self):
        if self.embeddings is None:
            from backend.retriever.embeddings import get_embeddings
            self.embeddings = get_embeddings()
        return self.embeddings

    def score(self, question: str, answer: str) -> Optional[float]:
        """Score entre 0 et 1, ou None si la mesure n'a pas pu être faite (erreur technique)."""
        canned = FinanceBenchJudge.detect_canned_response(answer)
        if canned == "ERROR":
            return None
        if canned == "REFUSAL":
            return 0.0
        try:
            content = self._get_llm().invoke(
                RELEVANCY_PROMPT.format(answer=_strip_citations(answer), n=self.n)
            ).content
            evasive, generated = parse_generated(content, self.n)
            if evasive:
                return 0.0
            if not generated:
                return None
            vectors = self._get_embeddings().embed_documents([question] + generated)
        except Exception as e:
            if is_rate_limit(e):
                raise  # rejoué par l'appelant (backoff), pas une mesure manquante
            return None
        sims = [_cosine(vectors[0], v) for v in vectors[1:]]
        return round(max(0.0, sum(sims) / len(sims)), 4)
