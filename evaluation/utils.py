import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langsmith import Client
from backend.retriever.builder import RetrieverBuilder


def load_dataset(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset introuvable: {path}")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


def log_to_langsmith(name: str, summary: Dict, inputs: Dict):
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("⚠️  LANGSMITH_API_KEY non défini, skip logging")
        return None

    project = os.getenv("LANGSMITH_PROJECT", "agentic_rag_multi_agent_evals")
    try:
        client = Client()
        run_id = uuid.uuid4()
        # LangSmith n'accepte que: tool, chain, llm, retriever, embedding, prompt, parser.
        # "evaluation" était refusé avec un 422 et le logging ne fonctionnait donc jamais.
        now = datetime.now(timezone.utc)
        client.create_run(
            id=run_id,
            name=name,
            run_type="chain",
            inputs=inputs,
            outputs=summary,
            project_name=project,
            # Sans end_time, le run resterait affiché « en cours » indéfiniment.
            start_time=now,
            end_time=now,
        )
        # create_run met en file d'attente : sans flush, une erreur serveur passerait
        # inaperçue et le message de succès serait mensonger.
        client.flush()
        print(f"✓ Résultats envoyés à LangSmith (projet: {project})")
        return run_id
    except Exception as e:
        print(f"⚠️  Erreur LangSmith: {e}")
        return None


def index_chunks(tenant_id: str, chunks: List, pages_by_doc: Optional[Dict[str, List[str]]] = None,
                 force: bool = False):
    """
    Indexe des chunks déjà produits dans l'espace Qdrant `tenant_id`, document par document.
    Idempotent : un document dont les chunks n'ont pas changé n'est pas ré-embeddé.
    Retourne le TenantStore.
    """
    from backend.vectorstore import get_store

    tenant = get_store().for_tenant(tenant_id)
    by_source: Dict[str, List] = {}
    for chunk in chunks:
        by_source.setdefault(str(chunk.metadata.get("source")), []).append(chunk)
    for source, doc_chunks in by_source.items():
        digest = hashlib.sha256(source.encode())
        for chunk in doc_chunks:
            digest.update(chunk.page_content.encode())
        file_hash = digest.hexdigest()
        if tenant.has_document(file_hash) and not force:
            continue
        pages = (pages_by_doc or {}).get(source, [])
        tenant.add_document(file_hash, source, doc_chunks, pages, enforce_quota=False)
    return tenant


def build_retriever_from_chunks(chunks: List, tenant_id: str,
                                pages_by_doc: Optional[Dict[str, List[str]]] = None):
    """
    Construit le retriever à partir de chunks déjà produits, sans repasser par l'OCR.

    L'évaluation FinanceBench pré-calcule ses chunks (OCR page par page + metadata de page)
    dans une phase séparée, les indexe dans un espace Qdrant dédié au jeu de documents, et
    réutilise ici exactement la même chaîne de retrieval que la production :
    Qdrant (BM25 sparse + dense) -> ParentChild -> MultiQuery -> Rerank Bedrock.
    """
    tenant = index_chunks(tenant_id, chunks, pages_by_doc)
    return RetrieverBuilder().build_hybrid_retriever(tenant)


# Résilience aux rate limits : implémentation côté backend (utilisée aussi par le
# workflow et les agents), ré-exportée ici pour les scripts d'évaluation.
from backend.utils.resilience import (  # noqa: E402,F401
    call_with_backoff,
    is_rate_limit,
    retry_after,
    root_cause,
)
