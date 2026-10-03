"""Le PageStore : grep / read_page / toc sur des pages OCR, sans appel réseau."""
from backend.retriever.page_store import PageStore, doc_label

PAGES = {
    "/tmp/AMD_2022_10K.pdf": [
        "# Item 1. Business\nAMD designs semiconductors.",
        "## Segment Information\nData Center segment revenue was $6,043 million.\n| Segment | 2022 |\n| Data Center | 6,043 |",
        "Provision for income taxes (122)\nEffective tax rate 21.6%",
    ],
    "BOEING_2022_10K": ["Boeing page one", "Effective tax rate 0.6%"],
}


def test_labels_and_resolution():
    store = PageStore(PAGES)
    assert doc_label("/tmp/AMD_2022_10K.pdf") == "AMD_2022_10K"
    assert store.documents() == ["AMD_2022_10K", "BOEING_2022_10K"]
    assert store.page_count("AMD_2022_10K") == 3
    assert store.source_of("AMD_2022_10K") == "/tmp/AMD_2022_10K.pdf"
    # libellé exact, chemin, sous-chaîne insensible à la casse
    assert store.resolve("AMD_2022_10K") == "AMD_2022_10K"
    assert store.resolve("/tmp/AMD_2022_10K.pdf") == "AMD_2022_10K"
    assert store.resolve("boeing") == "BOEING_2022_10K"
    assert store.resolve("2022_10K") is None  # ambigu


def test_grep_is_exhaustive_and_page_indexed():
    store = PageStore(PAGES)
    out = store.grep("effective tax rate")
    assert out["total"] == 2
    assert {(h["doc"], h["page"]) for h in out["hits"]} == {("AMD_2022_10K", 2), ("BOEING_2022_10K", 1)}
    scoped = store.grep("effective tax rate", doc="AMD_2022_10K")
    assert scoped["total"] == 1 and scoped["hits"][0]["page"] == 2
    # 0 résultat = le terme est absent, c'est une information
    assert store.grep("gross margin", doc="AMD_2022_10K") == {"hits": [], "total": 0}
    # regex invalide -> repli littéral, pas d'exception
    assert store.grep("(122", doc="AMD_2022_10K")["total"] == 1
    assert "error" in store.grep("x", doc="UNKNOWN")


def test_read_page_returns_whole_page_and_checks_bounds():
    store = PageStore(PAGES)
    page = store.read_page("AMD_2022_10K", 1)
    assert "| Data Center | 6,043 |" in page["text"] and page["n_pages"] == 3
    assert "error" in store.read_page("AMD_2022_10K", 3)
    assert "error" in store.read_page("AMD_2022_10K", -1)
    assert "error" in store.read_page("NOPE", 0)
    truncated = store.read_page("AMD_2022_10K", 1, max_chars=20)
    assert truncated["truncated"] and truncated["text"].endswith("[page tronquée]")


def test_page_document_carries_chunk_compatible_metadata():
    store = PageStore(PAGES)
    doc = store.page_document("AMD_2022_10K", 2)
    assert doc.metadata == {
        "source": "/tmp/AMD_2022_10K.pdf", "doc_name": "AMD_2022_10K", "page": 2, "origin": "read_page",
    }
    assert store.page_document("AMD_2022_10K", 99) is None


def test_toc_lists_markdown_headers_with_pages():
    store = PageStore(PAGES)
    toc = store.toc("AMD_2022_10K")
    assert [(e["page"], e["level"], e["title"]) for e in toc["entries"]] == [
        (0, 1, "Item 1. Business"), (1, 2, "Segment Information"),
    ]


def test_read_pages_spans_consecutive_pages_with_a_cap():
    store = PageStore(PAGES)
    out = store.read_pages("AMD_2022_10K", 1, 2)
    assert out["pages"] == [1, 2]
    assert "=== p. 2 ===" in out["text"] and "=== p. 3 ===" in out["text"]
    assert store.read_pages("AMD_2022_10K", 2, 1)["pages"] == [1, 2]  # bornes inversées
    assert store.read_pages("AMD_2022_10K", 0, 10, max_pages=2)["pages"] == [0, 1]  # plafond
    assert "error" in store.read_pages("AMD_2022_10K", 7, 9)
    docs = store.page_documents("AMD_2022_10K", 1, 2)
    assert [d.metadata["page"] for d in docs] == [1, 2] and all(d.metadata["origin"] == "read_page" for d in docs)


TEN_K = {
    "ACME_2022_10K": [
        "# INDEX\nConsolidated Balance Sheets ..... 3",
        "# PART II\n# **ITEM 8. FINANCIAL STATEMENTS**\n## ***Risk: a heading that is not a landmark***",
        "# Consolidated Balance Sheets\n| | 2022 |\n| Cash | 4,835 |",
        "## NOTE 5 – Income Taxes\nThe provision for income taxes was $122 million.\n# Note 5 – Income Taxes",
    ],
}


def test_outline_keeps_structural_landmarks_cleaned_and_deduplicated():
    out = PageStore(TEN_K).outline("ACME_2022_10K")
    assert out["n_pages"] == 4
    assert [(e["page"], e["title"]) for e in out["entries"]] == [
        (1, "PART II"), (1, "ITEM 8. FINANCIAL STATEMENTS"),
        (2, "Consolidated Balance Sheets"), (3, "NOTE 5 – Income Taxes"),
    ]
    # sans repère financier : repli sur les en-têtes de niveau 1
    assert [e["title"] for e in PageStore(PAGES).outline("AMD_2022_10K")["entries"]] == ["Item 1. Business"]
    assert "error" in PageStore(PAGES).outline("NOPE")


def test_find_sections_matches_titles_only_with_all_words():
    store = PageStore(TEN_K)
    out = store.find_sections("ACME_2022_10K", "balance sheet")
    # le sommaire (p. 1) mentionne le bilan dans le texte, pas dans un titre : écarté
    assert [(h["page"], h["title"]) for h in out["hits"]] == [(2, "Consolidated Balance Sheets")]
    assert out["hits"][0]["excerpt"].startswith("| | 2022 |")
    assert store.find_sections("ACME_2022_10K", "income taxes")["total"] == 2
    assert store.find_sections("ACME_2022_10K", "cash flows")["total"] == 0
    assert "error" in store.find_sections("ACME_2022_10K", "  ")
    assert "error" in store.find_sections("NOPE", "x")
