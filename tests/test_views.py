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
from conftest import make_doc  # noqa: E402


class FakeProcessor:
    def __init__(self):
        self.pages = {}

    def process(self, files):
        self.pages = {f.name: ["page 1", "page 2"] for f in files}
        return [make_doc(f"chunk de {f.name}") for f in files]


class FakeBuilder:
    def build_hybrid_retriever(self, chunks):
        return ("retriever", len(chunks))


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(views, "DocumentProcessor", FakeProcessor)
    monkeypatch.setattr(views, "RetrieverBuilder", FakeBuilder)
    monkeypatch.setattr(views, "sessions", {})
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
    assert r.json() == {"draft_answer": "réponse [1]", "verification_report": "rapport", "citations": [{"n": 1}]}
    assert seen == {"question": "quel CA ?", "retriever": ("retriever", 1), "pages": 2}


def test_sessions_are_isolated(api, monkeypatch):
    monkeypatch.setattr(views._workflow, "full_pipeline", lambda **k: {"draft_answer": "", "verification_report": ""})
    api.post("/api/upload-file", {"file": SimpleUploadedFile("a.pdf", b"a"), "session_id": "alice"})
    r = _post_json(api, "/api/process-question", {"question": "q", "session_id": "bob"})
    assert r.status_code == 400


def test_upload_rejects_unsupported_types_and_missing_file(api):
    r = api.post("/api/upload-file", {"file": SimpleUploadedFile("x.exe", b"MZ"), "session_id": "s"})
    assert r.status_code == 400 and "non supporté" in r.json()["error"]
    assert api.post("/api/upload-file", {"session_id": "s"}).status_code == 400


def test_load_file_reads_from_the_examples_dir(api):
    r = _post_json(api, "/api/load-file", {"file_name": "rapport.pdf", "session_id": "s"})
    assert r.status_code == 200
    assert r.json()["filename"] == "rapport.pdf"
    assert views.sessions["s"]["retriever"] == ("retriever", 1)


@pytest.mark.parametrize("name", ["../secret.md", "/etc/hosts", "", "sub/../../secret.md"])
def test_load_file_refuses_paths_outside_the_examples_dir(api, name):
    r = _post_json(api, "/api/load-file", {"file_name": name, "session_id": "s"})
    assert r.status_code == 400
    assert "s" not in views.sessions


def test_pipeline_failure_is_a_json_500(api, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("Mistral indisponible")

    monkeypatch.setattr(views._workflow, "full_pipeline", boom)
    api.post("/api/upload-file", {"file": SimpleUploadedFile("a.pdf", b"a"), "session_id": "s"})
    r = _post_json(api, "/api/process-question", {"question": "q", "session_id": "s"})
    assert r.status_code == 500 and "indisponible" in r.json()["error"]


def test_get_on_api_routes_is_rejected(api):
    assert api.get("/api/process-question").status_code == 405
