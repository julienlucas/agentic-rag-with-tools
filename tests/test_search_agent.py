"""L'agent de recherche à outils, avec un LLM factice qui émet des appels d'outils."""
from langchain_core.messages import AIMessage

from backend.agents.search_agent import SearchAgent
from backend.config.settings import settings
from backend.retriever.page_store import PageStore
from conftest import FakeRetriever, make_doc


class FakeToolLLM:
    """Renvoie une séquence d'AIMessage (avec ou sans tool_calls) ; bind_tools est un no-op."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.invocations = []

    def bind_tools(self, tools):
        self.tool_names = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.invocations.append(list(messages))
        if not self.turns:
            return AIMessage(content="terminé")
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            return AIMessage(content=turn)
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": f"call{i:05d}"} for i, (name, args) in enumerate(turn)
        ])


PAGES = {"AMD_2022_10K": ["Item 1", "Balance sheet\nCash 4,835\nCurrent liabilities 6,369", "Item 3"]}


def _head(n=None):
    n = settings.CORRECTIVE_PROTECT_TOP if n is None else n
    return [make_doc(f"initial-{i}", source="AMD_2022_10K", page=i) for i in range(n)]


def test_read_pages_come_first_then_reranked_search_results_after_protected_head():
    llm = FakeToolLLM([
        [("grep", {"pattern": "current liabilities", "doc": "AMD_2022_10K"})],
        [("read_page", {"doc": "AMD_2022_10K", "page": 2})],
        [("search", {"query": "AMD cash and short-term investments 2022"})],
        "J'ai trouvé le bilan p. 2.",
    ])
    retriever = FakeRetriever(default=[make_doc("search-hit", source="AMD_2022_10K", page=1),
                                       make_doc("initial-0", source="AMD_2022_10K", page=0)])
    agent = SearchAgent(llm=llm)
    head = _head()
    out = agent.run("quick ratio d'AMD ?", retriever, head + [make_doc("tail")],
                    page_store=PageStore(PAGES))

    assert llm.tool_names == ["search", "grep", "read_page", "open_document", "navigate"]
    assert out["tool_calls"] == 3
    docs = out["documents"]
    # tête intouchable, dans le même ordre
    assert [d.page_content for d in docs[:len(head)]] == [d.page_content for d in head]
    extras = docs[len(head):]
    assert extras[0].metadata["origin"] == "read_page" and extras[0].metadata["page"] == 1
    assert "Current liabilities 6,369" in extras[0].page_content
    # le résultat de search vient après la page lue ; le doublon de la tête est écarté
    assert [d.page_content for d in extras[1:]] == ["search-hit"]
    assert len(extras) <= settings.CORRECTIVE_EXTRA_DOCS
    # trace lisible pour le rapport de vérification
    assert out["queries"] == [
        'grep: "current liabilities" [AMD_2022_10K]',
        "read_page: AMD_2022_10K p. 2",
        'search: "AMD cash and short-term investments 2022"',
    ]
    # les résultats d'outils sont bien renvoyés au modèle (ToolMessage après chaque appel)
    tool_msgs = [m for m in llm.invocations[-1] if m.type == "tool"]
    assert len(tool_msgs) == 3
    assert "p. 2: Current liabilities 6,369" in tool_msgs[0].content
    assert tool_msgs[1].content.startswith("=== AMD_2022_10K, p. 2 ===")
    assert "[1] (AMD_2022_10K, p. 2) search-hit" in tool_msgs[2].content


def test_budget_caps_tool_calls_and_still_returns_head():
    calls = [[("grep", {"pattern": f"t{i}", "doc": "AMD_2022_10K"})] for i in range(10)]
    llm = FakeToolLLM(calls)
    agent = SearchAgent(llm=llm)
    agent.max_tool_calls = 2
    out = agent.run("q", FakeRetriever(), _head(3), page_store=PageStore(PAGES))
    assert out["tool_calls"] == 2
    assert [d.page_content for d in out["documents"]] == ["initial-0", "initial-1", "initial-2"]


def test_no_tool_calls_keeps_context_unchanged():
    llm = FakeToolLLM(["Rien à chercher."])
    out = SearchAgent(llm=llm).run("q", FakeRetriever(), _head(4), page_store=PageStore(PAGES))
    assert out == {"documents": _head(4), "queries": [], "tool_calls": 0} or (
        [d.page_content for d in out["documents"]] == [d.page_content for d in _head(4)]
        and out["queries"] == [] and out["tool_calls"] == 0
    )


def test_search_uses_document_scope_when_the_model_names_a_doc():
    class Scoped(FakeRetriever):
        def __init__(self):
            super().__init__(default=[make_doc("scoped", source="AMD_2022_10K", page=0)])
            self.scopes = []

        def invoke_with_scope(self, query, sources):
            self.scopes.append(sources)
            return self.invoke(query)

    r = Scoped()
    llm = FakeToolLLM([[("search", {"query": "litigation", "doc": "amd"})], "ok"])
    SearchAgent(llm=llm).run("q", r, [], scope=["ALL"], page_store=PageStore({"/x/AMD_2022_10K.pdf": ["p"]}))
    assert r.scopes == [["/x/AMD_2022_10K.pdf"]]  # le libellé est résolu vers la source


def test_tools_degrade_gracefully_without_page_store():
    llm = FakeToolLLM([[("grep", {"pattern": "x"}), ("read_page", {"doc": "AMD_2022_10K", "page": 1})], "ok"])
    out = SearchAgent(llm=llm).run("q", FakeRetriever(), _head(2), page_store=None)
    tool_msgs = [m for m in llm.invocations[-1] if m.type == "tool"]
    assert all("indisponible" in m.content for m in tool_msgs)
    assert len(out["documents"]) == 2


def test_read_page_can_span_pages_and_note_is_returned():
    llm = FakeToolLLM([
        [("read_page", {"doc": "AMD_2022_10K", "page": 2, "end_page": 3})],
        "NOTE : bilan p. 2-3, current liabilities 6 369.",
    ])
    out = SearchAgent(llm=llm).run("q", FakeRetriever(), _head(2), page_store=PageStore(PAGES))
    extras = out["documents"][2:]
    assert [d.metadata["page"] for d in extras] == [1, 2]
    assert out["queries"] == ["read_page: AMD_2022_10K p. 2-3"]
    assert out["note"] == "NOTE : bilan p. 2-3, current liabilities 6 369."
    tool_msg = [m for m in llm.invocations[-1] if m.type == "tool"][0]
    assert "=== AMD_2022_10K, p. 2 ===" in tool_msg.content and "=== AMD_2022_10K, p. 3 ===" in tool_msg.content


def test_generator_with_tools_numbers_fetched_passages_and_answers_in_place():
    """Le modèle qui cherche est celui qui répond : les passages ramenés par ses outils
    reçoivent un numéro qu'il voit dans le résultat, et la réponse finale les cite."""
    from backend.agents.research_agent import ResearchAgent

    llm = FakeToolLLM([
        [("search", {"query": "AMD balance sheet", "doc": "AMD_2022_10K"})],
        [("read_page", {"doc": "AMD_2022_10K", "page": 2})],
        "Quick ratio = 4,835 / 6,369 [3][4]",
    ])
    agent = ResearchAgent.__new__(ResearchAgent)  # sans clients Mistral
    agent.model = llm
    agent.model_tools = llm
    retriever = FakeRetriever(default=[make_doc("search-hit", source="AMD_2022_10K", page=1),
                                       make_doc("initial-0", source="AMD_2022_10K", page=0)])
    initial = [make_doc("initial-0", source="AMD_2022_10K", page=0), make_doc("initial-1", source="AMD_2022_10K", page=1)]
    out = agent.generate_with_tools("quick ratio ?", initial, retriever, page_store=PageStore(PAGES),
                                    max_tool_calls=3, relevance="PARTIAL")

    assert out["draft_answer"] == "Quick ratio = 4,835 / 6,369 [3][4]"
    assert out["tool_calls"] == 2
    # initiaux [1][2] conservés, search-hit -> [3] (initial-0 déjà [1]), page lue -> [4]
    assert [c["n"] for c in out["citations"]] == [1, 2, 3, 4]
    assert [d.page_content for d in out["documents"]][:3] == ["initial-0", "initial-1", "search-hit"]
    assert out["documents"][3].metadata["origin"] == "read_page"
    tool_msgs = [m for m in llm.invocations[-1] if m.type == "tool"]
    assert "[3] (AMD_2022_10K, p. 2) search-hit" in tool_msgs[0].content
    assert "[1] (AMD_2022_10K, p. 1) initial-0" in tool_msgs[0].content  # doublon : garde son numéro
    assert tool_msgs[1].content.startswith("=== [4] AMD_2022_10K, p. 2 ===")
    # le prompt système porte les règles et le guide des outils, avec le budget
    system = llm.invocations[0][0]
    assert "RÈGLES STRICTES" in system.content and "3 appels d'outils au maximum" in system.content
    assert "jugé le contexte initial PARTIEL" in system.content  # le verdict du vérificateur, en indice


def test_generator_with_tools_forces_an_answer_when_budget_is_exhausted():
    from backend.agents.research_agent import ResearchAgent

    # Deux appels dans le même tour, budget d'un seul : le second est refusé, puis le
    # modèle est rappelé sans outils pour conclure.
    llm = FakeToolLLM([
        [("grep", {"pattern": "a", "doc": "AMD_2022_10K"}), ("grep", {"pattern": "b", "doc": "AMD_2022_10K"})],
        "réponse forcée",
    ])
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.model = llm
    agent.model_tools = llm
    out = agent.generate_with_tools("q", [make_doc("x")], FakeRetriever(), page_store=PageStore(PAGES), max_tool_calls=1)
    assert out["tool_calls"] == 1
    assert out["draft_answer"] == "réponse forcée"
    # le second appel a été refusé avec un message de budget, puis conclusion forcée
    budget_msgs = [m for m in llm.invocations[-1] if m.type == "tool" and "Budget" in m.content]
    assert budget_msgs


def test_block_list_content_is_flattened_to_text():
    """Mistral renvoie parfois `content` en liste de blocs : 2 questions perdues sur un run."""
    from backend.agents.search_agent import message_text

    assert message_text(AIMessage(content="plain")) == "plain"
    assert message_text(AIMessage(content=[{"type": "text", "text": "a "}, {"type": "text", "text": "b"}])) == "a b"
    assert message_text(AIMessage(content=[])) == ""


def test_open_document_and_navigate_point_to_pages_without_adding_passages():
    pages = {"AMD_2022_10K": [
        "# PART II\n# ITEM 8. FINANCIAL STATEMENTS",
        "# Consolidated Balance Sheets\nCash 4,835\nCurrent liabilities 6,369",
        "# NOTE 4 – Segment Reporting",
    ]}
    llm = FakeToolLLM([
        [("open_document", {})],
        [("open_document", {"doc": "amd"})],
        [("navigate", {"doc": "AMD_2022_10K", "section": "balance sheet"})],
        [("navigate", {"doc": "AMD_2022_10K", "section": "cash flows"})],
        "Bilan p. 2.",
    ])
    out = SearchAgent(llm=llm).run("q", FakeRetriever(), _head(2), page_store=PageStore(pages))
    tool_msgs = [m.content for m in llm.invocations[-1] if m.type == "tool"]
    assert tool_msgs[0] == "- AMD_2022_10K (3 pages)"
    assert tool_msgs[1].splitlines() == [
        "AMD_2022_10K — 3 pages", "p. 1 : PART II", "p. 1 : ITEM 8. FINANCIAL STATEMENTS",
        "p. 2 : Consolidated Balance Sheets", "p. 3 : NOTE 4 – Segment Reporting",
    ]
    assert tool_msgs[2].startswith("p. 2 : Consolidated Balance Sheets — Cash 4,835")
    assert tool_msgs[3].startswith("Aucun titre ne contient « cash flows »")
    assert out["queries"] == [
        "open_document", "open_document: amd",
        'navigate: "balance sheet" [AMD_2022_10K]', 'navigate: "cash flows" [AMD_2022_10K]',
    ]
    # navigation seulement : aucun passage ajouté après la tête
    assert len(out["documents"]) == 2
