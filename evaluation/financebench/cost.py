"""
Coût d'un run FinanceBench, à partir des tokens réellement facturés.

Le runner compte les tokens de chaque appel LLM (callback LangChain, par modèle) et les
unités de recherche Cohere (`billed_units.search_units` de chaque réponse Rerank). Ce module
applique la grille tarifaire publique et écrit le total dans `financebench_summary.json`,
pour que « relancer l'éval coûte X € » soit un chiffre mesuré, pas une estimation.

Non compté : les embeddings de requête pendant le run (quelques dizaines de tokens par
recherche, Mistral Embed à 0,10 $/M : négligeable) et la préparation (OCR + embeddings des
documents, payée une fois — voir `prep_cost_usd`).
"""

from typing import Dict, Optional

# Grille publique La Plateforme (https://mistral.ai/pricing/api) et Cohere
# (https://cohere.com/pricing), relevée le 5 septembre 2026, en dollars.
# Claude : grille Anthropic de la Claude API, relevée le 28 septembre 2026.
# ⚠️ Depuis le passage à Bedrock (2 oct. 2026), Claude et Cohere Embed v4 sont facturés par
# AWS : grille Bedrock (https://aws.amazon.com/bedrock/pricing/) NON relevée ici — les
# montants Claude ci-dessous sont ceux de la grille Anthropic, à vérifier. Le rerank reste chez Cohere.
PRICES_USD = {
    # $ par million de tokens (entrée, sortie)
    "claude-sonnet-5": {"input_per_m": 2.00, "output_per_m": 10.00},
    "claude-sonnet-4-6": {"input_per_m": 3.00, "output_per_m": 15.00},
    "claude-haiku-4-5": {"input_per_m": 1.00, "output_per_m": 5.00},
    "mistral-large": {"input_per_m": 0.50, "output_per_m": 1.50},
    "mistral-small": {"input_per_m": 0.15, "output_per_m": 0.60},
    "mistral-embed": {"input_per_m": 0.10, "output_per_m": 0.0},
    # $ par page OCRisée
    "mistral-ocr": {"per_page": 4.0 / 1000},
    # $ par unité de recherche (une requête, jusqu'à 100 documents)
    "rerank-v4.0-pro": {"per_search": 0.0025},
}

# Taux de la BCE au 4 septembre 2026 (api.frankfurter.dev). À mettre à jour avec la grille.
USD_TO_EUR = 0.86
PRICES_DATE = "2026-09-05"


def _price_for(model: str) -> Optional[Dict[str, float]]:
    # Identifiants Bedrock : « eu.anthropic.claude-sonnet-5 » -> « claude-sonnet-5 ».
    name = (model or "").lower()
    for prefix in ("eu.", "us.", "global.", "apac."):
        name = name.removeprefix(prefix)
    name = name.removeprefix("anthropic.")
    for prefix, price in PRICES_USD.items():
        if name.startswith(prefix):
            return price
    return None


def compute_cost(usage_by_model: Dict[str, Dict[str, int]],
                 rerank_usage: Optional[Dict[str, float]] = None) -> Dict:
    """
    `usage_by_model` : la sortie de `UsageMetadataCallbackHandler.usage_metadata`
    ({modèle: {input_tokens, output_tokens, total_tokens}}).
    `rerank_usage` : {"calls": n, "search_units": u} (voir `backend.retriever.builder`).
    """
    lines, total = {}, 0.0
    unpriced = []
    for model, usage in sorted((usage_by_model or {}).items()):
        price = _price_for(model)
        tokens_in = int(usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("output_tokens") or 0)
        if price is None or "input_per_m" not in price:
            unpriced.append(model)
            cost = None
        else:
            cost = (tokens_in * price["input_per_m"] + tokens_out * price["output_per_m"]) / 1e6
            total += cost
        lines[model] = {"input_tokens": tokens_in, "output_tokens": tokens_out,
                        "usd": None if cost is None else round(cost, 4)}

    rerank = rerank_usage or {}
    units = float(rerank.get("search_units") or 0)
    rerank_cost = units * PRICES_USD["rerank-v4.0-pro"]["per_search"]
    total += rerank_cost
    lines["cohere-rerank"] = {"calls": int(rerank.get("calls") or 0),
                              "search_units": round(units, 1), "usd": round(rerank_cost, 4)}

    return {
        "total_usd": round(total, 2),
        "total_eur": round(total * USD_TO_EUR, 2),
        "usd_to_eur": USD_TO_EUR,
        "prices_date": PRICES_DATE,
        "by_model": lines,
        "unpriced_models": unpriced,
        "note": "Tokens facturés par appel (génération, sous-agents, juge LLM) + unités de "
                "recherche Cohere. Hors embeddings de requête (négligeables) et préparation.",
    }


def prep_cost_usd(n_pages: int, n_chunk_tokens: int) -> float:
    """Coût de la préparation : OCR des pages + embedding des chunks (payé une fois)."""
    return (n_pages * PRICES_USD["mistral-ocr"]["per_page"]
            + n_chunk_tokens * PRICES_USD["mistral-embed"]["input_per_m"] / 1e6)


def format_cost(cost: Dict) -> str:
    parts = []
    for model, u in cost.get("by_model", {}).items():
        if "search_units" in u:
            parts.append(f"{model}: {u['calls']} appels, {u['search_units']:.0f} unités, {u['usd']:.2f} $")
        else:
            usd = "—" if u["usd"] is None else f"{u['usd']:.2f} $"
            parts.append(f"{model}: {u['input_tokens']:,} in / {u['output_tokens']:,} out, {usd}")
    head = (f"Coût du run: {cost['total_usd']:.2f} $ ≈ {cost['total_eur']:.2f} € "
            f"(grille du {cost['prices_date']}, 1 $ = {cost['usd_to_eur']} €)")
    return head + "\n  " + "\n  ".join(parts)
