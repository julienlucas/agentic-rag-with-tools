"""
Accès page par page aux documents OCRisés — les outils de navigation de l'agent de recherche.

Le retrieval par chunks ne montre jamais une page entière au modèle : un tableau de 1 200
caractères y est coupé, un signe négatif se perd, une note de segment n'est pas remontée.
Le PageStore garde le markdown de chaque page tel que l'OCR l'a produit, et expose des
opérations façon système de fichiers :

- grep(pattern, doc)      : où apparaît un terme, page par page — matching littéral, exhaustif.
                            Un grep qui renvoie 0 résultat sur 260 pages fonde une réponse
                            négative (« cette métrique n'est pas publiée »), là où « pas dans
                            les 15 passages que j'ai vus » ne prouve rien.
- read_page(doc, page)    : la page entière, tableau compris.
- toc(doc)                : les en-têtes markdown avec leur page, pour naviguer par section.
- outline(doc)            : le plan — PART, ITEM, états financiers, notes —, ce que l'outil
                            open montre au modèle.
- find_sections(doc, s)   : les en-têtes dont l'intitulé correspond à `s` (outil navigate).

Les documents sont identifiés par leur libellé (nom de fichier sans extension), le même que
celui qu'emploie le routage par document.
"""
import os
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document


# Repères d'un rapport financier (10-K, 10-Q, rapport annuel), pour le plan d'un document.
_LANDMARK = re.compile(
    r"^(part\s+[ivx]+\b|item\s+\d+[a-z]?\b|note\s+\d+\b|notes?\s+to\s+(the\s+)?consolidated"
    r"|consolidated\s+(statements?|balance\s+sheets?)\b|report\s+of\s+independent)",
    re.IGNORECASE,
)


def _clean_title(title: str) -> str:
    """Intitulé d'en-tête sans l'emphase markdown (**, ***, _) ni les espaces superflus."""
    return " ".join(re.sub(r"[*_`]+", "", title).split())[:120]


def _normalize(text: str) -> str:
    return _clean_title(text).lower().replace("&", "and")


def doc_label(source: str) -> str:
    """Nom lisible d'une source (basename sans extension), identique à document_router.source_label."""
    return os.path.splitext(os.path.basename(str(source)))[0]


class PageStore:
    """Pages OCR par document, adressables par (libellé, numéro de page zero-indexed)."""

    def __init__(self, pages_by_source: Dict[str, List[str]]):
        self._pages: Dict[str, List[str]] = {}
        self._source: Dict[str, str] = {}
        for source, pages in pages_by_source.items():
            label = doc_label(source)
            self._pages[label] = list(pages or [])
            self._source[label] = str(source)

    # --- inventaire ---------------------------------------------------------

    def documents(self) -> List[str]:
        return list(self._pages.keys())

    def page_count(self, doc: str) -> int:
        return len(self._pages.get(doc, []))

    def source_of(self, doc: str) -> Optional[str]:
        """Source d'origine (chemin de fichier ou nom), pour le périmètre du retriever."""
        return self._source.get(doc)

    def resolve(self, doc: Optional[str]) -> Optional[str]:
        """Accepte un libellé exact, un chemin, ou une sous-chaîne insensible à la casse."""
        if not doc:
            return None
        if doc in self._pages:
            return doc
        label = doc_label(doc)
        if label in self._pages:
            return label
        low = label.lower()
        matches = [d for d in self._pages if low in d.lower() or d.lower() in low]
        return matches[0] if len(matches) == 1 else None

    # --- outils ---------------------------------------------------------------

    def grep(self, pattern: str, doc: Optional[str] = None, max_hits: int = 20,
             context_chars: int = 160) -> Dict:
        """
        Cherche `pattern` (regex, insensible à la casse ; repli littéral si la regex est
        invalide) dans un document ou dans tous. Retourne les occurrences page par page.
        """
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)

        target = self.resolve(doc) if doc else None
        if doc and target is None:
            return {"hits": [], "total": 0, "error": f"document inconnu: {doc}", "documents": self.documents()}
        docs = [target] if target else self.documents()

        hits, total = [], 0
        for d in docs:
            for page_idx, text in enumerate(self._pages[d]):
                for line in text.splitlines():
                    if rx.search(line):
                        total += 1
                        if len(hits) < max_hits:
                            snippet = " ".join(line.split())
                            if len(snippet) > context_chars:
                                # Centre l'extrait sur la première occurrence.
                                m = rx.search(snippet)
                                start = max(0, (m.start() if m else 0) - context_chars // 3)
                                snippet = ("…" if start else "") + snippet[start:start + context_chars] + "…"
                            hits.append({"doc": d, "page": page_idx, "line": snippet})
        return {"hits": hits, "total": total}

    def read_page(self, doc: str, page: int, max_chars: int = 8000) -> Dict:
        """Le markdown complet d'une page (zero-indexed), tronqué au-delà de max_chars."""
        target = self.resolve(doc)
        if target is None:
            return {"error": f"document inconnu: {doc}", "documents": self.documents()}
        pages = self._pages[target]
        try:
            page = int(page)
        except (TypeError, ValueError):
            return {"error": f"page invalide: {page!r}"}
        if page < 0 or page >= len(pages):
            return {"error": f"page {page} hors limites (0-{len(pages) - 1})", "doc": target}
        text = pages[page]
        truncated = len(text) > max_chars
        return {
            "doc": target, "page": page, "n_pages": len(pages),
            "text": text[:max_chars] + ("\n… [page tronquée]" if truncated else ""),
            "truncated": truncated,
        }

    def read_pages(self, doc: str, start: int, end: int, max_pages: int = 3,
                   max_chars: int = 12000) -> Dict:
        """
        Plusieurs pages consécutives (zero-indexed, bornes incluses), pour les tableaux à
        cheval sur deux pages — le cas du tableau des flux de trésorerie de PepsiCo, où
        l'agent lisait la p. 65 quand la preuve continuait p. 64. Plafonné à max_pages.
        """
        target = self.resolve(doc)
        if target is None:
            return {"error": f"document inconnu: {doc}", "documents": self.documents()}
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            return {"error": f"pages invalides: {start!r}-{end!r}"}
        if end < start:
            start, end = end, start
        n = len(self._pages[target])
        start, end = max(0, start), min(n - 1, end)
        if start >= n:
            return {"error": f"page {start} hors limites (0-{n - 1})", "doc": target}
        end = min(end, start + max_pages - 1)
        pages = list(range(start, end + 1))
        budget = max_chars // len(pages)
        parts = []
        for p in pages:
            text = self._pages[target][p]
            parts.append(f"=== p. {p + 1} ===\n" + text[:budget] + ("\n… [page tronquée]" if len(text) > budget else ""))
        return {"doc": target, "pages": pages, "n_pages": n, "text": "\n\n".join(parts)}

    def toc(self, doc: str, max_entries: int = 120) -> Dict:
        """En-têtes markdown (#, ##, ###) avec leur page."""
        target = self.resolve(doc)
        if target is None:
            return {"error": f"document inconnu: {doc}", "documents": self.documents()}
        entries = []
        for page_idx, text in enumerate(self._pages[target]):
            for line in text.splitlines():
                m = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
                if m:
                    entries.append({"page": page_idx, "level": len(m.group(1)), "title": m.group(2)[:120]})
        return {"doc": target, "n_pages": len(self._pages[target]),
                "entries": entries[:max_entries], "total": len(entries)}

    def outline(self, doc: str, max_entries: int = 60) -> Dict:
        """
        Le plan d'un document : ses repères structurels (PART, ITEM, états financiers
        consolidés, notes, rapport d'audit) avec leur page. Le niveau des en-têtes OCR n'est
        pas fiable — un 10-K en compte 200 à 550, facteurs de risque compris —, on filtre donc
        par intitulé. Sans repère reconnu (document non financier), repli sur les en-têtes
        de niveau 1.
        """
        result = self.toc(doc, max_entries=100000)
        if "error" in result:
            return result
        seen, landmarks = set(), []
        for e in result["entries"]:
            title = _clean_title(e["title"])
            key = title.lower()
            if key in seen or not _LANDMARK.match(title):
                continue
            seen.add(key)
            landmarks.append({"page": e["page"], "title": title})
        if len(landmarks) < 3:
            landmarks = [{"page": e["page"], "title": _clean_title(e["title"])}
                         for e in result["entries"] if e["level"] == 1]
        return {"doc": result["doc"], "n_pages": result["n_pages"],
                "entries": landmarks[:max_entries], "total": len(landmarks)}

    def find_sections(self, doc: str, section: str, max_hits: int = 12,
                      excerpt_chars: int = 200) -> Dict:
        """
        Les en-têtes dont l'intitulé contient tous les mots de `section` (insensible à la
        casse et à la mise en forme), avec leur page et le début du texte qui suit. Là où
        grep sur « cash flows » renvoie le sommaire, les renvois et les notes, ceci ne
        renvoie que les titres.
        """
        target = self.resolve(doc)
        if target is None:
            return {"error": f"document inconnu: {doc}", "documents": self.documents()}
        words = re.findall(r"\w+", _normalize(section))
        if not words:
            return {"error": f"section vide: {section!r}"}
        hits, total = [], 0
        for page_idx, text in enumerate(self._pages[target]):
            lines = text.splitlines()
            for i, line in enumerate(lines):
                m = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
                if not m:
                    continue
                title = _clean_title(m.group(1))
                norm = _normalize(title)
                if not all(w in norm for w in words):
                    continue
                total += 1
                if len(hits) < max_hits:
                    after = " ".join(" ".join(lines[i + 1:]).split())
                    hits.append({"page": page_idx, "title": title, "excerpt": after[:excerpt_chars]})
        return {"doc": target, "hits": hits, "total": total}

    # --- vers le contexte du modèle ------------------------------------------

    def page_documents(self, doc: str, start: int, end: int, max_pages: int = 3) -> List[Document]:
        """Une page = un Document, pour une plage de pages (voir read_pages)."""
        result = self.read_pages(doc, start, end, max_pages=max_pages)
        if "error" in result:
            return []
        docs = [self.page_document(result["doc"], p) for p in result["pages"]]
        return [d for d in docs if d is not None]

    def page_document(self, doc: str, page: int, max_chars: int = 8000) -> Optional[Document]:
        """La page en Document, avec les mêmes metadata que les chunks (source, doc_name, page)."""
        result = self.read_page(doc, page, max_chars=max_chars)
        if "error" in result:
            return None
        return Document(
            page_content=result["text"],
            metadata={
                "source": self._source[result["doc"]],
                "doc_name": result["doc"],
                "page": result["page"],
                "origin": "read_page",
            },
        )
