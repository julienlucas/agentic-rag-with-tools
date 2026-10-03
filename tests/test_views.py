"""L'API HTTP, avec le client de test Django : OCR, index et pipeline sont remplacés par des faux."""
import json
import os

import django
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.test import Client  # noqa: E402

from backend import views  # noqa: E402
from backend.config.settings import settings  # noqa: E402
from conftest import make_doc, make_store  # noqa: E402


class FakeProcessor:
    calls = 0

    def __init__(self):
        self.pages = {}

    def process(self, files):
        FakeProcessor.calls += 1
        self.pages = {f.name: ["page 1", "page 2"] for f in files}
        return [make_doc(f"chunk de {f.name}", source=f.name) for f in files]


class FakeBuilder:
    def build_hybrid_retriever(self, tenant):
        return ("retriever", tenant.tenant_id, len(tenant.sources()))


@pytest.fixture
def store(monkeypatch):
    s = make_store()
    monkeypatch.setattr(views, "get_store", lambda: s)
    monkeypatch.setattr(views, "_page_cache", {})
    return s


@pytest.fixture
def api(monkeypatch, tmp_path, store):
    FakeProcessor.calls = 0
    monkeypatch.setattr(views, "DocumentProcessor", FakeProcessor)
    monkeypatch.setattr(views, "RetrieverBuilder", FakeBuilder)
    monkeypatch.setattr(settings, "EXAMPLES_DIR", str(tmp_path))
    (tmp_path / "rapport.pdf").write_bytes(b"%PDF-1.4 exemple")
    (tmp_path.parent / "secret.md").write_text("hors du dossier d'exemples")
    return Client()


def _post_json(client, url, payload):
    return client.post(url, data=json.dumps(payload), content_type="application/json")


def test_question_before_any_document_is_a_400(api):
    r = _post_json(api, "/api/process-question", {"question": "q", "session_id": "s"})
    assert r.status_code == 400
    assert "Aucun document" in r.json()["error"]


def test_upload_then_question_goes_through_the_pipeline(api, monkeypatch):
    seen = {}

    def full_pipeline(question, retriever, page_store=None):
        seen.update(question=question, retriever=retriever, pages=page_store.page_count("rapport"))
        return {"draft_answer": "réponse [1]", "verification_report": "rapport", "citations": [{"n": 1}]}

    monkeypatch.setattr(views._workflow, "full_pipeline", full_pipeline)
    up = api.post("/api/upload-file", {"file": SimpleUploadedFile("rapport.pdf", b"%PDF-1.4"), "session_id": "s"})
    assert up.status_code == 200 and up.json()["chunks_count"] == 1

    r = _post_json(api, "/api/process-question", {"question": "  quel CA ?  ", "session_id": "s"})
    assert r.status_code == 200
    assert r.json() == {"draft_answer": "réponse [1]", "verification_report": "rapport", "citations": [{"n": 1}],
                        "feedback_token": None}  # votes désactivés sans LANGSMITH_API_KEY
    assert seen == {"question": "quel CA ?", "retriever": ("retriever", "s", 1), "pages": 2}


def test_sessions_are_isolated(api, monkeypatch):
    monkeypatch.setattr(views._workflow, "full_pipeline", lambda **k: {"draft_answer": "", "verification_report": ""})
    api.post("/api/upload-file", {"file": SimpleUploadedFile("a.pdf", b"a"), "session_id": "alice"})
    r = _post_json(api, "/api/process-question", {"question": "q", "session_id": "bob"})
    assert r.status_code == 400


def test_upload_rejects_unsupported_types_and_missing_file(api):
    r = api.post("/api/upload-file", {"file": SimpleUploadedFile("x.exe", b"MZ"), "session_id": "s"})
    assert r.status_code == 400 and "non supporté" in r.json()["error"]
    assert api.post("/api/upload-file", {"session_id": "s"}).status_code == 400


def test_load_file_reads_from_the_examples_dir(api, store):
    r = _post_json(api, "/api/load-file", {"file_name": "rapport.pdf", "session_id": "s"})
    assert r.status_code == 200
    assert r.json()["filename"] == "rapport.pdf"
    assert [os.path.basename(src) for src in store.for_tenant("s").sources()] == ["rapport.pdf"]


@pytest.mark.parametrize("name", ["../secret.md", "/etc/hosts", "", "sub/../../secret.md"])
def test_load_file_refuses_paths_outside_the_examples_dir(api, store, name):
    r = _post_json(api, "/api/load-file", {"file_name": name, "session_id": "s"})
    assert r.status_code == 400
    assert store.for_tenant("s").documents() == []


def test_documents_accumulate_in_the_space_and_are_not_reprocessed(api, store):
    api.post("/api/upload-file", {"file": SimpleUploadedFile("a.pdf", b"a"), "session_id": "s"})
    api.post("/api/upload-file", {"file": SimpleUploadedFile("b.pdf", b"b"), "session_id": "s"})
    again = api.post("/api/upload-file", {"file": SimpleUploadedFile("a.pdf", b"a"), "session_id": "s"})
    assert again.status_code == 200 and again.json()["message"] == "Document déjà indexé"
    assert FakeProcessor.calls == 2  # pas d'OCR pour un fichier déjà indexé
    listed = api.get("/api/documents", {"session_id": "s"}).json()["documents"]
    assert sorted(d["name"] for d in listed) == ["a.pdf", "b.pdf"]


def test_delete_document_and_space(api):
    api.post("/api/upload-file", {"file": SimpleUploadedFile("a.pdf", b"a"), "session_id": "s"})
    api.post("/api/upload-file", {"file": SimpleUploadedFile("b.pdf", b"b"), "session_id": "s"})
    first = api.get("/api/documents", {"session_id": "s"}).json()["documents"][0]
    assert _post_json(api, "/api/delete-document", {"session_id": "other", "file_hash": first["file_hash"]}).status_code == 404
    assert _post_json(api, "/api/delete-document", {"session_id": "s", "file_hash": first["file_hash"]}).status_code == 200
    assert len(api.get("/api/documents", {"session_id": "s"}).json()["documents"]) == 1
    assert _post_json(api, "/api/delete-space", {"session_id": "s"}).status_code == 200
    assert api.get("/api/documents", {"session_id": "s"}).json()["documents"] == []


def test_invalid_session_id_is_a_400(api):
    r = _post_json(api, "/api/process-question", {"question": "q", "session_id": "../x y"})
    assert r.status_code == 400
    r = api.post("/api/upload-file", {"file": SimpleUploadedFile("a.pdf", b"a"), "session_id": "a b"})
    assert r.status_code == 400


def test_pipeline_failure_is_a_json_500(api, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("Mistral indisponible")

    monkeypatch.setattr(views._workflow, "full_pipeline", boom)
    api.post("/api/upload-file", {"file": SimpleUploadedFile("a.pdf", b"a"), "session_id": "s"})
    r = _post_json(api, "/api/process-question", {"question": "q", "session_id": "s"})
    assert r.status_code == 500 and "indisponible" in r.json()["error"]


def test_get_on_api_routes_is_rejected(api):
    assert api.get("/api/process-question").status_code == 405
