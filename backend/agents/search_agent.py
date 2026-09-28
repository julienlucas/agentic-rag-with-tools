"""
Outils de recherche façon système de fichiers, et l'agent de recherche qui les utilise.

Trois outils, fermés sur le retriever et le PageStore d'une question :

- search(query, doc)              : le retrieval hybride + rerank du pipeline (c'est sa force à
                                    cette échelle, on ne le remplace pas — on le rend itératif).
- grep(pattern, doc)              : occurrences littérales page par page, exhaustif. Un 0 résultat
                                    sur 260 pages fonde une réponse négative.
- read_page(doc, page, end_page)  : une page entière, ou 2-3 pages consécutives pour un tableau
                                    à cheval — ce qu'un chunk de 1 200 caractères ne montre jamais.

Deux usages :

1. `build_tools()` — le modèle de GÉNÉRATION reçoit ces outils directement (voir
   ResearchAgent.generate_with_tools) : il cherche et répond dans la même conversation, comme
   dans l'Agentic Search de Mistral. C'est le mode par défaut (settings.GENERATOR_TOOLS_ENABLED).

2. `SearchAgent` — un agent séparé collecte la preuve manquante puis la transmet au générateur,
   avec sa note de synthèse. Mode conditionnel, déclenché par le vérificateur de pertinence
   (settings.CORRECTIVE_MODE = "agent"). Conservé pour comparaison.

Dans les deux cas, le contrat avec le reste du pipeline est le même : les RESEARCH_TOP_K
passages initiaux sont intouchables, les outils ne peuvent qu'AJOUTER après eux.
"""
import hashlib
from typing import Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ..config.settings import settings
from ..retriever.page_store import PageStore, doc_label
from ..utils.logging import logger
from ..utils.resilience import is_rate_limit
from .corrective_retrieval import CorrectiveRetrieval, _find_reranker
from .reasoning_llm import reasoning_llm


def _key(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def message_text(message) -> str:
    """
    Texte d'un AIMessage. `content` est une str, ou une liste de blocs ({"type": "text", ...})
    selon la réponse de l'API : le run FinanceBench du 4 sept. 2026 (soir) a perdu 2 questions
    sur un `'list' object has no attribute 'strip'`.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def _locator(doc: Document) -> str:
    meta = doc.metadata or {}
    name = doc_label(meta.get("doc_name") or meta.get("source") or "document")
    page = meta.get("page")
    return f"{name}, p. {int(page) + 1}" if page is not None else name


def build_tools(
    retriever,
    page_store: Optional[PageStore],
    scope,
    trace: Dict,
    on_documents: Optional[Callable[[List[Document]], List[int]]] = None,
):
    """
    Construit search / grep / read_page fermés sur le retriever, le PageStore et la trace.

    `trace` reçoit : calls (libellés lisibles), search_results (listes de Documents),
    read_pages (Documents de pages lues).

    `on_documents(docs) -> [numéros]` : si fourni, chaque Document ramené par search ou
    read_page est enregistré dans le contexte numéroté du générateur, et le numéro attribué
    est affiché dans le résultat de l'outil pour que le modèle puisse le citer [n].
    """
    found_docs: List[List[Document]] = trace.setdefault("search_results", [])
    read_pages: List[Document] = trace.setdefault("read_pages", [])
    calls: List[str] = trace.setdefault("calls", [])

    def _scope_for(doc: Optional[str]):
        if doc and page_store is not None:
            label = page_store.resolve(doc)
            if label:
                return [page_store.source_of(label)]
        if doc:
            return [doc]
        return scope

    def _numbers(docs: List[Document]) -> List[Optional[int]]:
        if on_documents is None or not docs:
            return [None] * len(docs)
        return list(on_documents(docs))

    @tool
    def search(query: str, doc: Optional[str] = None) -> str:
        """Recherche sémantique + lexicale dans les documents indexés. `query` : une phrase
        courte en langage naturel, dans le vocabulaire des rapports financiers (pas
        d'opérateur booléen). `doc` : restreindre à un document (optionnel)."""
        calls.append(f'search: "{query}"' + (f" [{doc}]" if doc else ""))
        try:
            target = _scope_for(doc)
            if target is not None and hasattr(retriever, "invoke_with_scope"):
                results = retriever.invoke_with_scope(query, target)
            else:
                results = retriever.invoke(query)
        except Exception as e:
            if is_rate_limit(e):
                raise
            logger.warning(f"search a échoué: {e}")
            return "Erreur de recherche, réessayez avec une autre formulation."
        results = list(results or [])[:8]
        found_docs.append(results)
        if not results:
            return "Aucun résultat."
        numbers = _numbers(results)
        lines = []
        for i, (d, n) in enumerate(zip(results, numbers), start=1):
            tag = f"[{n}]" if n is not None else f"[{i}]"
            excerpt = " ".join(d.page_content.split())[:400]
            lines.append(f"{tag} ({_locator(d)}) {excerpt}")
        return "\n".join(lines)

    @tool
    def grep(pattern: str, doc: Optional[str] = None) -> str:
        """Occurrences littérales d'un terme (regex insensible à la casse), page par page.
        Exhaustif : 0 résultat signifie que le terme est absent du document. `doc` :
        restreindre à un document (optionnel, recommandé)."""
        calls.append(f'grep: "{pattern}"' + (f" [{doc}]" if doc else ""))
        if page_store is None:
            return "grep indisponible (pas de pages OCR pour ce corpus)."
        result = page_store.grep(pattern, doc=doc)
        if result.get("error"):
            return f"{result['error']}. Documents : {', '.join(result.get('documents', []))}"
        if not result["hits"]:
            return f"0 occurrence de « {pattern} »."
        lines = [f"{h['doc']} p. {h['page'] + 1}: {h['line']}" for h in result["hits"]]
        more = result["total"] - len(result["hits"])
        if more > 0:
            lines.append(f"… et {more} autres occurrences (affinez le motif ou le document).")
        return "\n".join(lines)

    @tool
    def read_page(doc: str, page: int, end_page: Optional[int] = None) -> str:
        """Lit une page entière d'un document (numéro tel qu'affiché par search et grep, à
        partir de 1). Pour un tableau à cheval sur plusieurs pages, donnez `end_page` :
        les pages `page` à `end_page` sont lues ensemble (3 au plus)."""
        span = f"{page}-{end_page}" if end_page is not None and end_page != page else f"{page}"
        calls.append(f"read_page: {doc} p. {span}")
        if page_store is None:
            return "read_page indisponible (pas de pages OCR pour ce corpus)."
        try:
            start = int(page) - 1
            end = int(end_page) - 1 if end_page is not None else start
        except (TypeError, ValueError):
            return f"page invalide: {page!r}"
        documents = page_store.page_documents(doc, start, end)
        if not documents:
            result = page_store.read_page(doc, start)
            return result.get("error", "page introuvable")
        new = [d for d in documents if all(_key(d) != _key(p) for p in read_pages)]
        read_pages.extend(new)
        numbers = _numbers(documents)
        parts = []
        for d, n in zip(documents, numbers):
            tag = f"[{n}] " if n is not None else ""
            parts.append(f"=== {tag}{_locator(d)} ===\n{d.page_content}")
        return "\n\n".join(parts)

    return [search, grep, read_page]


def run_tool_loop(llm, tools, messages: List, max_tool_calls: int, final_llm=None) -> Dict:
    """
    Boucle d'appels d'outils bornée. `messages` est complété en place.
    Retourne {"tool_calls": n, "final": AIMessage}. Quand le budget est épuisé, un dernier
    tour SANS outils (`final_llm`, ou `llm` s'il n'est pas fourni) force une conclusion.
    """
    tool_map = {t.name: t for t in tools}
    bound = llm.bind_tools(tools)
    n_calls = 0
    final: Optional[AIMessage] = None
    for _turn in range(max_tool_calls + 1):
        ai: AIMessage = bound.invoke(messages)
        messages.append(ai)
        tool_calls = list(getattr(ai, "tool_calls", None) or [])
        if not tool_calls:
            final = ai
            break
        for call in tool_calls:
            if n_calls >= max_tool_calls:
                messages.append(ToolMessage(
                    content="Budget d'appels épuisé : conclus avec ce que tu as.",
                    tool_call_id=call["id"],
                ))
                continue
            n_calls += 1
            fn = tool_map.get(call["name"])
            try:
                output = fn.invoke(call["args"]) if fn else f"outil inconnu: {call['name']}"
            except Exception as e:
                if is_rate_limit(e):
                    raise
                logger.warning(f"outil {call['name']} a levé {e}")
                output = f"Erreur outil: {type(e).__name__}"
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        if n_calls >= max_tool_calls:
            final = (final_llm or llm).invoke(messages)
            messages.append(final)
            break
    if final is None:
        final = (final_llm or llm).invoke(messages)
        messages.append(final)
    return {"tool_calls": n_calls, "final": final}


SYSTEM_PROMPT = """Tu es un analyste financier. Une première recherche documentaire n'a PAS suffi à répondre à la question ; tu disposes d'outils pour trouver la preuve manquante dans les documents indexés.

Documents disponibles :
{documents}

Ce que le modèle a déjà sous les yeux (passages initiaux, ils resteront dans son contexte) :
{initial}

Méthode :
1. Identifie ce qui manque précisément (un chiffre, une ligne comptable, une section).
2. `search` pour une recherche sémantique en langage naturel dans le vocabulaire des rapports annuels ("provision for income taxes", "Legal Proceedings", "segment information").
3. `grep` pour vérifier qu'un terme existe (ou n'existe pas) dans un document, et savoir sur quelle page.
4. `read_page` pour lire une page entière quand un résultat pointe vers un tableau ou une note : les tableaux sont coupés dans les extraits, jamais dans la page. Si le tableau continue sur la page suivante, lis les deux (`end_page`).
5. Arrête-toi dès que tu as lu la ou les pages qui portent la preuve — les pages lues seront transmises telles quelles au modèle qui répond. Tu as {budget} appels d'outils au maximum.

Si la question porte sur une métrique qui doit être calculée (ratio, marge, variation), cherche les lignes de base du calcul (ex. quick ratio : cash, short-term investments, receivables, current liabilities — dans le bilan consolidé).

Quand tu as terminé, écris une NOTE courte pour le modèle qui répondra : ce que tu as trouvé, sur quelle(s) page(s), les chiffres exacts avec leur signe, ou ce que tu as constaté absent (et sur quelle base — un grep sans résultat, par exemple)."""


class SearchAgent:
    """Agent séparé : cherche la preuve manquante, la transmet au générateur avec une note."""

    def __init__(self, llm=None):
        self.llm = llm
        self.max_tool_calls = settings.CORRECTIVE_MAX_TOOL_CALLS

    def _get_llm(self):
        if self.llm is None:
            self.llm = reasoning_llm()
        return self.llm

    @staticmethod
    def _describe_initial(docs: List[Document], limit: int) -> str:
        lines = []
        for i, d in enumerate(docs[:limit], start=1):
            lines.append(f"[{i}] ({_locator(d)}) {' '.join(d.page_content.split())[:160]}…")
        return "\n".join(lines) or "(aucun)"

    def run(
        self,
        question: str,
        retriever,
        current_docs: List[Document],
        scope: Optional[List[str]] = None,
        page_store: Optional[PageStore] = None,
    ) -> Dict:
        """
        Cherche la preuve manquante avec les outils, puis assemble le contexte :
        tête protégée (inchangée) + pages lues + résultats de recherche reclassés.

        Retourne {"documents", "queries" (trace), "tool_calls", "note"} — la note est la
        conclusion de l'agent, transmise au générateur pour qu'il sache ce qui a été trouvé
        et pourquoi ces pages sont là.
        """
        protect = max(0, settings.CORRECTIVE_PROTECT_TOP)
        extra_budget = max(0, settings.CORRECTIVE_EXTRA_DOCS)
        head = current_docs[:protect]
        head_keys = {_key(d) for d in head}

        trace: Dict = {}
        tools = build_tools(retriever, page_store, scope, trace)

        documents = (page_store.documents() if page_store is not None else None) or sorted({
            doc_label((d.metadata or {}).get("doc_name") or (d.metadata or {}).get("source") or "")
            for d in current_docs
        } - {""})
        system = SYSTEM_PROMPT.format(
            documents="\n".join(f"- {d}" for d in documents) or "- (inconnus)",
            initial=self._describe_initial(current_docs, protect),
            budget=self.max_tool_calls,
        )
        messages = [SystemMessage(content=system), HumanMessage(content=f"Question : {question}")]
        loop = run_tool_loop(self._get_llm(), tools, messages, self.max_tool_calls)
        n_calls = loop["tool_calls"]
        note = message_text(loop["final"]).strip() if n_calls else ""

        # --- Assemblage du contexte : la tête ne bouge pas, on ajoute après. ---
        extras: List[Document] = []
        seen = set(head_keys)
        for page in trace.get("read_pages", []):
            if len(extras) >= extra_budget:
                break
            k = _key(page)
            if k not in seen:
                extras.append(page)
                seen.add(k)

        remaining = extra_budget - len(extras)
        if remaining > 0 and trace.get("search_results"):
            merged = [
                d for d in CorrectiveRetrieval._merge(trace["search_results"], top_n=30)
                if _key(d) not in seen
            ]
            reranker = _find_reranker(retriever)
            if reranker is not None and merged:
                merged = reranker.rerank(question, merged, top_n=remaining)
            for d in merged[:remaining]:
                extras.append(d)
                seen.add(_key(d))

        logger.info(
            f"SearchAgent: {n_calls} appel(s) d'outils, {len(trace.get('read_pages', []))} page(s) lue(s), "
            f"{len(extras)} passage(s) ajouté(s) après les {len(head)} initiaux"
        )
        return {
            "documents": head + extras,
            "queries": list(trace.get("calls", [])),
            "tool_calls": n_calls,
            "note": note,
        }
