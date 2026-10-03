"""Votes 👍/👎 : jeton signé, route /api/feedback, et les fonctions pures de evaluation/feedback/."""
import json
import os
import uuid

import django
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402

from backend import feedback, views  # noqa: E402
from backend.config.settings import settings  # noqa: E402
from evaluation.feedback.collect import cited_passages, example_from_vote, parse_classification  # noqa: E402
from evaluation.feedback.replay import outcome, replayable, summarize  # noqa: E402
from test_views import api, store, _post_json  # noqa: E402,F401  (fixtures)


# --- jeton -----------------------------------------------------------------

def test_token_round_trip_is_bound_to_the_tenant():
    run_id = uuid.uuid4()
    token = feedback.issue_token(run_id, "alice")
    assert feedback.read_token(token, "alice") == run_id
    with pytest.raises(feedback.InvalidFeedback):
        feedback.read_token(token, "bob")
    with pytest.raises(feedback.InvalidFeedback):
        feedback.read_token(token[:-2] + "xx", "alice")


def test_feedback_id_is_deterministic():
    run_id = uuid.uuid4()
    assert feedback.feedback_id(run_id) == feedback.feedback_id(run_id) != run_id


# --- API -------------------------------------------------------------------

@pytest.fixture
def voting(api, monkeypatch):
    sent = []
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setattr(feedback, "send", lambda run_id, score, comment=None: sent.append((run_id, score, comment)))
    monkeypatch.setattr(views._workflow, "full_pipeline",
                        lambda **k: {"draft_answer": "réponse", "verification_report": "", "citations": []})
    api.post("/api/upload-file", {"file": SimpleUploadedFile("a.pdf", b"a"), "session_id": "alice"})
    return api, sent


def _ask(client, session="alice"):
    return _post_json(client, "/api/process-question", {"question": "q", "session_id": session}).json()


def test_vote_reaches_the_traced_run(voting):
    client, sent = voting
    token = _ask(client)["feedback_token"]
    r = _post_json(client, "/api/feedback",
                   {"feedback_token": token, "score": 0, "comment": " chiffre faux ", "session_id": "alice"})
    assert r.status_code == 200
    assert len(sent) == 1 and sent[0][1:] == (0, "chiffre faux")
    assert sent[0][0] == feedback.read_token(token, "alice")


def test_vote_from_another_session_is_refused(voting):
    client, sent = voting
    token = _ask(client)["feedback_token"]
    r = _post_json(client, "/api/feedback", {"feedback_token": token, "score": 1, "session_id": "bob"})
    assert r.status_code == 400 and not sent


@pytest.mark.parametrize("score", [2, -1, True, "1", None])
def test_vote_score_must_be_0_or_1(voting, score):
    client, sent = voting
    token = _ask(client)["feedback_token"]
    r = _post_json(client, "/api/feedback", {"feedback_token": token, "score": score, "session_id": "alice"})
    assert r.status_code == 400 and not sent


def test_vote_is_disabled_without_langsmith(api):
    r = _post_json(api, "/api/feedback", {"feedback_token": "x", "score": 1, "session_id": "alice"})
    assert r.status_code == 503


def test_langsmith_failure_is_a_502(voting, monkeypatch):
    client, _ = voting
    token = _ask(client)["feedback_token"]

    def down(*a, **k):
        raise RuntimeError("LangSmith indisponible")

    monkeypatch.setattr(feedback, "send", down)
    r = _post_json(client, "/api/feedback", {"feedback_token": token, "score": 1, "session_id": "alice"})
    assert r.status_code == 502


# --- collect.py ------------------------------------------------------------

def test_parse_classification_tolerates_markdown_and_unknown_causes():
    assert parse_classification("**CAUSE:** erreur_calcul\nRAISON: marge mal calculée") == (
        "erreur_calcul", "marge mal calculée")
    assert parse_classification("CAUSE: [Refus_A_Tort]\nRAISON: x")[0] == "refus_a_tort"
    assert parse_classification("CAUSE: inventée")[0] == "autre"
    assert parse_classification("") == ("autre", "")


def test_cited_passages_keeps_only_cited_numbers():
    citations = [{"n": 1, "locator": "A p.3", "excerpt": "un"}, {"n": 2, "locator": "A p.9", "excerpt": "deux"}]
    assert cited_passages("CA en hausse [2].", citations) == "[2] A p.9 — deux"
    assert cited_passages("sans citation", citations) == "(aucun)"


def test_example_from_vote_up_becomes_a_reference():
    ex = example_from_vote("r1", "q", {"draft_answer": "42 M€"}, {"tenant_id": "t", "documents": ["h"]},
                           1.0, None, "2026-10-03")
    assert ex["outputs"] == {"answer": "42 M€"}
    assert ex["metadata"]["vote"] == "up" and ex["metadata"]["documents"] == ["h"]
    assert "rejected_answer" not in ex["metadata"]


def test_example_from_vote_down_keeps_the_rejected_answer_out_of_outputs():
    ex = example_from_vote("r1", "q", {"draft_answer": "41 M€"}, {}, 0.0, "c'est 42", "2026-10-03",
                           "erreur_lecture", "mauvaise ligne")
    assert ex["outputs"] is None
    assert ex["metadata"]["rejected_answer"] == "41 M€"
    assert ex["metadata"]["failure_category"] == "erreur_lecture"
    assert ex["metadata"]["comment"] == "c'est 42"


# --- replay.py -------------------------------------------------------------

def test_replayable_needs_a_reference():
    examples = [{"outputs": {"answer": "x"}}, {"outputs": None}, {"outputs": {"answer": "  "}}, {}]
    assert replayable(examples) == [{"outputs": {"answer": "x"}}]


def test_only_a_lost_thumbs_up_is_a_regression():
    assert outcome("up", "CORRECT") == "OK"
    assert outcome("up", "INCORRECT") == "RÉGRESSION"
    assert outcome("up", "REFUSAL") == "RÉGRESSION"
    assert outcome("down", "CORRECT") == "CORRIGÉ"
    assert outcome("down", "INCORRECT") == "TOUJOURS FAUX"
    assert outcome("up", "ERROR") == "ERREUR"


def test_summarize_exit_codes():
    rows = lambda *o: [{"outcome": x} for x in o]  # noqa: E731
    assert summarize(rows("OK", "TOUJOURS FAUX"))["exit_code"] == 0
    assert summarize(rows("OK", "RÉGRESSION"))["exit_code"] == 1
    assert summarize(rows("SAUTÉ", "ERREUR"))["exit_code"] == 2
    assert summarize([])["exit_code"] == 2
