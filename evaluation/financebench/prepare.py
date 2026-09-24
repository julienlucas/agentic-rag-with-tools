# Prépare l'évaluation FinanceBench : télécharge le sous-ensemble, OCR page par page,
# chunke, met en cache, et génère le dataset.jsonl au format du projet.
#
# Usage:
#   uv run python evaluation/financebench/prepare.py
#   uv run python evaluation/financebench/prepare.py --docs AMD_2022_10K --force

import argparse
import base64
import hashlib
import json
import pickle
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mistralai import Mistral

from backend.config.settings import settings
from backend.document_processor.chunkers import ParentChildChunkingStrategy
from evaluation.utils import call_with_backoff

HERE = Path(__file__).resolve().parent
PDF_DIR = HERE / "pdfs"
CACHE_DIR = HERE / "cache"
CHROMA_DIR = HERE / "chroma"

GITHUB_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_URL = f"{GITHUB_RAW}/data/financebench_open_source.jsonl"

# Les 3 documents portant 7 questions chacun (21 questions), plus PepsiCo : 5 questions
# dont 2 `metrics-generated` à réponse numérique exacte (EBITDA, marge) = 26 questions.
DEFAULT_DOCS = [
    "AMD_2022_10K",
    "AMERICANEXPRESS_2022_10K",
    "BOEING_2022_10K",
    "PEPSICO_2022_10K",
]

# Jeu étendu : tous les documents portant au moins 3 questions (18 documents, 70 questions).
# À 26 questions, l'IC95 de l'accuracy fait ~35 points de large : un écart de 5 points entre
# deux versions n'y est pas mesurable. À 70, la largeur tombe vers ~20 points.
EXTENDED_MIN_QUESTIONS = 3


def extended_docs(questions: List[Dict], min_questions: int = EXTENDED_MIN_QUESTIONS) -> List[str]:
    """Documents du jeu étendu, triés : ceux qui portent au moins `min_questions` questions."""
    counts: Dict[str, int] = {}
    for q in questions:
        counts[q["doc_name"]] = counts.get(q["doc_name"], 0) + 1
    return sorted(d for d, n in counts.items() if n >= min_questions)


def dataset_path_for(docs: List[str], preset: str = "") -> Path:
    """
    Le dataset versionné (dataset.jsonl) porte les 26 questions du protocole publié. Tout
    autre jeu de documents écrit ailleurs : avant, `--docs AMD_2022_10K` le réécrivait avec
    7 questions, et le run suivant n'était plus comparable aux chiffres du README.
    """
    if sorted(docs) == sorted(DEFAULT_DOCS):
        return HERE / "dataset.jsonl"
    return HERE / ("dataset_extended.jsonl" if preset == "extended" else "dataset_custom.jsonl")


# Longueur cible des snippets gold découpés depuis evidence_text.
# Les evidence FinanceBench vont de 180 à 2400 caractères : un parent chunk (1200 car.)
# ne peut pas contenir 60% des tokens d'une evidence de 2400 car., donc _doc_relevance_flags
# ne matcherait jamais. On découpe en snippets de la taille d'un chunk enfant.
GOLD_SNIPPET_CHARS = 250


def _log(msg: str):
    print(f"[prepare] {msg}", flush=True)


def store_dir_for(docs: List[str]) -> Path:
    """
    Répertoire Chroma dédié à un jeu de documents précis.

    Indispensable : si on réutilisait une collection unique, un run lancé avec --docs sur un
    sous-ensemble verrait BM25 restreint au sous-ensemble mais la recherche vectorielle
    répondre depuis tous les documents indexés précédemment.
    """
    key = hashlib.sha256(",".join(sorted(docs)).encode()).hexdigest()[:12]
    return CHROMA_DIR / f"docs-{key}"


def load_cached_chunks(docs: List[str]) -> List:
    """Recharge les chunks mis en cache par la phase prepare."""
    chunks = []
    missing = []
    for doc_name in docs:
        cache_path = CACHE_DIR / f"{doc_name}.chunks.pkl"
        if not cache_path.exists():
            missing.append(doc_name)
            continue
        with open(cache_path, "rb") as f:
            chunks.extend(pickle.load(f))
    if missing:
        raise SystemExit(
            f"Chunks absents du cache pour: {', '.join(missing)}.\n"
            f"Lancez d'abord: uv run python evaluation/financebench/prepare.py "
            f"--docs {','.join(docs)}"
        )
    return chunks


def load_cached_pages(docs: List[str]) -> Dict[str, List[str]]:
    """
    Recharge le markdown OCR page par page mis en cache par la phase prepare.
    Alimente le PageStore (outils grep / read_page de l'agent de recherche).
    """
    pages_by_doc: Dict[str, List[str]] = {}
    for doc_name in docs:
        cached = _load_ocr_cache(CACHE_DIR / f"{doc_name}.ocr.json")
        if not cached:
            raise SystemExit(
                f"Cache OCR absent pour {doc_name}.\n"
                f"Lancez d'abord: uv run python evaluation/financebench/prepare.py --docs {doc_name}"
            )
        pages_by_doc[doc_name] = [cached.get(i, "") for i in range(max(cached) + 1)]
    return pages_by_doc


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Télécharge un fichier si absent."""
    if dest.exists() and not force and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Téléchargement {url} -> {dest.name}")
    with urllib.request.urlopen(url, timeout=180) as resp:
        data = resp.read()
    dest.write_bytes(data)
    _log(f"  {len(data) / 1e6:.1f} Mo")
    return dest


def load_questions(force: bool = False) -> List[Dict]:
    """Charge les 150 questions open-source de FinanceBench."""
    path = download(QUESTIONS_URL, CACHE_DIR / "financebench_open_source.jsonl", force)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pdf_page_count(pdf_path: Path) -> Optional[int]:
    """Nombre de pages lu localement, pour découper l'OCR en lots."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except Exception as e:
        _log(f"  comptage de pages impossible ({type(e).__name__}), OCR en un seul appel")
        return None


def _document_url(client: Mistral, pdf_path: Path) -> str:
    """Upload le PDF une fois et renvoie une URL signée (repli sur data URL base64)."""
    file_bytes = pdf_path.read_bytes()
    try:
        uploaded = call_with_backoff(
            lambda: client.files.upload(
                file={"file_name": pdf_path.name, "content": file_bytes},
                purpose="ocr",
            ),
            "l'upload du PDF", attempts=7, base_delay=15.0, log=_log,
        )
        signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
        return signed.url
    except Exception as e:
        _log(f"  upload indisponible ({type(e).__name__}: {e}), repli sur base64")
        return f"data:application/pdf;base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def _load_ocr_cache(cache_path: Path) -> Dict[int, str]:
    """Relit le cache OCR, éventuellement partiel. Tolère l'ancien format (liste)."""
    if not cache_path.exists():
        return {}
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    if isinstance(data, list):  # ancien format
        return {i: page for i, page in enumerate(data)}
    return {int(k): v for k, v in (data.get("pages") or {}).items()}


def _save_ocr_cache(cache_path: Path, pages: Dict[int, str], n_pages: Optional[int]):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_pages": n_pages, "pages": {str(k): v for k, v in sorted(pages.items())}}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def ocr_document(
    client: Mistral,
    pdf_path: Path,
    doc_name: str,
    force: bool = False,
    batch_size: int = 50,
    delay: float = 2.0,
) -> List[str]:
    """
    OCR un PDF via Mistral OCR en conservant le découpage par page.

    Contrairement à DocumentProcessor._process_file qui joint toutes les pages en un seul
    markdown, on garde une entrée par page pour pouvoir attacher le numéro de page en
    metadata (les evidence FinanceBench sont annotées par page, zero-indexed).

    L'OCR est découpé en lots et le cache est écrit après chaque lot : un rate limit ou une
    interruption ne fait pas perdre le travail déjà payé — relancer reprend où ça s'est arrêté.
    """
    cache_path = CACHE_DIR / f"{doc_name}.ocr.json"
    cached: Dict[int, str] = {} if force else _load_ocr_cache(cache_path)
    n_pages = _pdf_page_count(pdf_path)

    if cached and (n_pages is None or len(cached) >= n_pages):
        _log(f"{doc_name}: {len(cached)} pages depuis le cache OCR")
        return [cached.get(i, "") for i in range(max(cached) + 1)]

    if cached:
        _log(f"{doc_name}: reprise de l'OCR ({len(cached)}/{n_pages} pages déjà en cache)")
    else:
        _log(f"{doc_name}: OCR ({pdf_path.stat().st_size / 1e6:.1f} Mo, {n_pages or '?'} pages)...")

    start = time.time()
    url = _document_url(client, pdf_path)

    def ocr_call(page_indices: Optional[List[int]]):
        kwargs = {
            "model": settings.MODEL_OCR_ID,
            "document": {"type": "document_url", "document_url": url},
            "include_image_base64": False,  # inutile ici, et très lourd en réponse
        }
        if page_indices is not None:
            kwargs["pages"] = page_indices
        return client.ocr.process(**kwargs)

    if n_pages is None:
        # Pas de découpage possible : un seul appel, avec retry.
        response = call_with_backoff(lambda: ocr_call(None), f"l'OCR de {doc_name}",
                                     attempts=7, base_delay=15.0, log=_log)
        for page in response.pages:
            cached[int(page.index)] = page.markdown or ""
        n_pages = len(cached)
    else:
        todo = [i for i in range(n_pages) if i not in cached]
        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        for batch_no, batch in enumerate(batches, start=1):
            response = call_with_backoff(
                lambda b=batch: ocr_call(b),
                f"l'OCR de {doc_name} (pages {batch[0]}-{batch[-1]})",
                attempts=7, base_delay=15.0, log=_log,
            )
            for page in response.pages:
                # On se fie à page.index : l'API renvoie l'index absolu dans le document.
                cached[int(page.index)] = page.markdown or ""
            _save_ocr_cache(cache_path, cached, n_pages)
            _log(f"  {doc_name}: lot {batch_no}/{len(batches)} — {len(cached)}/{n_pages} pages")
            if batch_no < len(batches) and delay:
                time.sleep(delay)

    _save_ocr_cache(cache_path, cached, n_pages)
    pages = [cached.get(i, "") for i in range(n_pages)]
    total_chars = sum(len(p) for p in pages)
    _log(
        f"{doc_name}: {len(pages)} pages OCR en {time.time() - start:.0f}s "
        f"({total_chars / 1000:.0f}k caractères)"
    )
    return pages


def chunk_document(pages: List[str], doc_name: str, force: bool = False) -> List:
    """
    Chunke un document page par page en attachant source + page en metadata.

    On utilise explicitement ParentChildChunkingStrategy (recursive) plutôt que la factory
    get_chunking_strategy() : la stratégie sémantique embed les paragraphes par batchs de 16
    en série, ce qui prendrait plusieurs minutes par 10-K de 250 pages.
    """
    cache_path = CACHE_DIR / f"{doc_name}.chunks.pkl"
    if cache_path.exists() and not force:
        with open(cache_path, "rb") as f:
            chunks = pickle.load(f)
        _log(f"{doc_name}: {len(chunks)} chunks depuis le cache")
        return chunks

    chunker = ParentChildChunkingStrategy()
    all_chunks = []
    for page_idx, page_text in enumerate(pages):
        if not page_text.strip():
            continue
        metadata = {"source": doc_name, "doc_name": doc_name, "page": page_idx}
        all_chunks.extend(chunker.split(page_text, metadata))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(all_chunks, f)
    _log(f"{doc_name}: {len(all_chunks)} chunks (children) générés")
    return all_chunks


def _split_evidence(text: str, target: int = GOLD_SNIPPET_CHARS) -> List[str]:
    """
    Découpe une evidence longue en snippets ~250 caractères, en respectant les fins de phrase
    et les sauts de ligne (les evidence de 10-K sont souvent des tableaux multi-lignes).
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]

    # Découpe en unités atomiques : lignes puis phrases.
    units: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= target:
            units.append(line)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())

    snippets: List[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > target:
            snippets.append(current)
            current = unit
        else:
            current = candidate
    if current:
        snippets.append(current)

    # Les snippets trop courts (< 40 car.) matchent trop facilement n'importe quel chunk.
    return [s for s in snippets if len(s) >= 40] or [text[:target]]


def _keywords_from_answer(answer: str) -> List[str]:
    """
    Extrait des mots-clés de la réponse gold pour context_hit_rate : nombres/montants
    et groupes nominaux capitalisés (noms de segments, de postes comptables...).
    """
    answer = answer or ""
    keywords = []
    # Montants et pourcentages : $1,577.00 / 16% / 1.57
    keywords.extend(re.findall(r"\$?\d[\d,]*\.?\d*\s*%?", answer)[:6])
    # Groupes nominaux capitalisés (Data Center, Customer deposits...)
    keywords.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", answer)[:4])

    seen, out = set(), []
    for k in keywords:
        # Retire la ponctuation de bord ("737," -> "737", "2023." -> "2023")
        k = k.strip().strip(".,;:")
        # Une année seule n'est pas discriminante dans un 10-K, elle apparaît partout.
        if re.fullmatch(r"(19|20)\d{2}", k):
            continue
        if len(k) >= 2 and k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)
    return out


def build_dataset(questions: List[Dict], docs: List[str]) -> List[Dict]:
    """Convertit les questions FinanceBench au format dataset du projet."""
    selected = [q for q in questions if q["doc_name"] in set(docs)]
    rows = []
    for q in selected:
        gold_passages, gold_pages = [], []
        for ev in q.get("evidence", []):
            gold_passages.extend(_split_evidence(ev.get("evidence_text", "")))
            page = ev.get("evidence_page_num")
            ev_doc = ev.get("evidence_doc_name") or q["doc_name"]
            if page is not None and str(page).strip() != "":
                gold_pages.append([ev_doc, int(page)])

        rows.append({
            "id": q["financebench_id"],
            "doc_name": q["doc_name"],
            "company": q.get("company"),
            "question": q["question"].strip(),
            "expected_answer": (q.get("answer") or "").strip(),
            "justification": (q.get("justification") or "").strip(),
            "answer_keywords": _keywords_from_answer(q.get("answer", "")),
            "gold_passages": gold_passages,
            "gold_pages": gold_pages,
            "question_type": q.get("question_type"),
            "question_reasoning": q.get("question_reasoning"),
        })

    # Groupé par type de question, pour la ventilation du résumé.
    order = {"metrics-generated": 0, "domain-relevant": 1, "novel-generated": 2}
    rows.sort(key=lambda r: (order.get(r["question_type"], 9), r["doc_name"], r["id"]))
    return rows


def write_dataset(rows: List[Dict], path: Path):
    """Écrit le JSONL, groupé par type de question sous des en-têtes `# Categorie (n)`."""
    by_type: Dict[str, List[Dict]] = {}
    for row in rows:
        by_type.setdefault(row["question_type"] or "other", []).append(row)

    docs = sorted({r["doc_name"] for r in rows})
    lines = [
        f"# FinanceBench (Patronus AI) - {len(rows)} questions sur {len(docs)} documents",
        f"# Documents: {', '.join(docs)}",
    ]
    for qtype, group in by_type.items():
        label = qtype.replace("-", " ").capitalize()
        lines.append(f"# {label} ({len(group)})")
        for row in group:
            lines.append(json.dumps(row, ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Dataset écrit: {path} ({len(rows)} questions)")


def build_vector_store(chunks: List, docs: List[str], force: bool = False):
    """
    Construit (et persiste) la collection Chroma pour éviter de ré-embedder à chaque run.
    ~5000 children sur 3 documents : plusieurs dizaines de secondes d'embeddings Mistral.
    """
    from langchain_community.vectorstores import Chroma
    from backend.retriever.embeddings import get_embeddings

    store_dir = store_dir_for(docs)
    if force and store_dir.exists():
        import shutil
        shutil.rmtree(store_dir)

    store_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()
    store = Chroma(
        persist_directory=str(store_dir),
        embedding_function=embeddings,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        collection_metadata={"hnsw:space": settings.VECTOR_SPACE},
    )
    existing = store._collection.count()
    if existing >= len(chunks) and not force:
        _log(f"Collection Chroma déjà peuplée ({existing} vecteurs), on la réutilise")
        return store

    if existing > 0:
        # Indexation précédente interrompue : repartir de zéro plutôt que créer des doublons.
        _log(f"Collection incomplète ({existing}/{len(chunks)}), reconstruction complète")
        store.delete_collection()
        store = Chroma(
            persist_directory=str(store_dir),
            embedding_function=embeddings,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            collection_metadata={"hnsw:space": settings.VECTOR_SPACE},
        )
        existing = 0

    _log(f"Embedding de {len(chunks)} chunks (collection actuelle: {existing})...")
    start = time.time()
    BATCH = 128
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        store.add_documents(batch)
        _log(f"  {min(i + BATCH, len(chunks))}/{len(chunks)} chunks embeddés")
    _log(f"Collection Chroma construite en {time.time() - start:.0f}s")
    return store


def main():
    parser = argparse.ArgumentParser(description="Prépare l'évaluation FinanceBench")
    parser.add_argument("--docs", default=",".join(DEFAULT_DOCS),
                        help="Documents à ingérer, séparés par des virgules, ou 'extended' "
                             f"(tous les documents à {EXTENDED_MIN_QUESTIONS}+ questions : 18 documents, 70 questions)")
    parser.add_argument("--force", action="store_true", help="Ignore les caches et tout reconstruit")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="N'construit pas la collection Chroma (elle sera bâtie au 1er run)")
    parser.add_argument("--ocr-batch", type=int, default=50,
                        help="Pages par appel OCR. Baisser (ex. 20) si rate limits répétés")
    parser.add_argument("--ocr-delay", type=float, default=2.0,
                        help="Pause en secondes entre deux lots OCR")
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.force)
    _log(f"{len(questions)} questions FinanceBench chargées")

    preset = args.docs.strip() if args.docs.strip() == "extended" else ""
    docs = extended_docs(questions) if preset else [d.strip() for d in args.docs.split(",") if d.strip()]
    _log(f"Documents: {', '.join(docs)}")

    rows = build_dataset(questions, docs)
    if not rows:
        raise SystemExit(f"Aucune question trouvée pour: {docs}")
    dataset_path = dataset_path_for(docs, preset)
    write_dataset(rows, dataset_path)

    client = Mistral(api_key=settings.MISTRALAI_API_KEY)
    all_chunks = []
    for doc_name in docs:
        pdf_path = download(f"{GITHUB_RAW}/pdfs/{doc_name}.pdf", PDF_DIR / f"{doc_name}.pdf", args.force)
        pages = ocr_document(client, pdf_path, doc_name, args.force,
                             batch_size=args.ocr_batch, delay=args.ocr_delay)
        all_chunks.extend(chunk_document(pages, doc_name, args.force))

    _log(f"Total: {len(all_chunks)} chunks sur {len(docs)} documents")

    if not args.skip_embeddings:
        build_vector_store(all_chunks, docs, args.force)

    _log("Préparation terminée. Lancer maintenant:")
    if dataset_path.name == "dataset.jsonl":
        _log("  uv run python evaluation/financebench/run_financebench_eval.py --mode both")
    else:
        _log(f"  uv run python evaluation/financebench/run_financebench_eval.py --mode both "
             f"--dataset {dataset_path.relative_to(ROOT_DIR)} --out-dir evaluation/financebench/outputs_{dataset_path.stem}")


if __name__ == "__main__":
    main()
