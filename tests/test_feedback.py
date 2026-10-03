"""Votes 👍/👎 : jeton signé, route /api/feedback, rangement dans le dataset et rejeu."""
import os
import uuid

import django
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402

from backend import feedback, feedback_dataset, views  # noqa: E402
from backend.config.settings import settings  # noqa: E402
from backend.feedback_dataset import (  # noqa: E402
    TraceNotReady, add_vote, cited_passages, example_from_vote, parse_classification,
)
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
    monkeypatch.setattr(feedback_dataset, "collect_in_background",
                        lambda run_id, score, comment: sent.append(("dataset", run_id, score)))
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
    run_id = feedback.read_token(token, "alice")
    assert sent == [(run_id, 0, "chiffre faux"), ("dataset", run_id, 0)]  # puis rangé en arrière-plan


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


# --- feedback_dataset.py ---------------------------------------------------

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


# --- add_vote / arrière-plan ----------------------------------------------

class FakeRun:
    def __init__(self, run_id, done=True):
        self.id = run_id
        self.inputs = {"question": "quel CA ?"}
        self.outputs = {"draft_answer": "42 M€ [1]", "citations": [{"n": 1, "locator": "p.3", "excerpt": "CA"}]} if done else None
        self.end_time = "fin" if done else None
        self.extra = {"metadata": {"tenant_id": "alice", "documents": ["h1"]}}


class FakeLangSmith:
    def __init__(self, runs):
        self.runs, self.examples, self.feedback = runs, [], []

    def has_dataset(self, dataset_name):
        return True

    def read_dataset(self, dataset_name):
        return type("D", (), {"id": "ds"})()

    def list_examples(self, dataset_id, metadata=None, limit=None):
        return [e for e in self.examples if not metadata or e["metadata"]["run_id"] == metadata["run_id"]]

    def read_run(self, run_id):
        from langsmith.utils import LangSmithNotFoundError
        if run_id not in self.runs:
            raise LangSmithNotFoundError(run_id)
        return self.runs[run_id]

    def create_example(self, dataset_id, source_run_id, **ex):
        self.examples.append(ex)

    def create_feedback(self, run_id, key, value, comment):
        self.feedback.append((key, value))


class FakeJudge:
    def invoke(self, prompt):
        return type("R", (), {"content": "CAUSE: erreur_lecture\nRAISON: mauvaise ligne"})()


def test_add_vote_down_is_classified_and_stored_once():
    client = FakeLangSmith({"r1": FakeRun("r1")})
    ex = add_vote(client, "r1", 0, "c'est 41", "2026-10-03", llm=FakeJudge())
    assert ex["metadata"]["failure_category"] == "erreur_lecture"
    assert client.feedback == [("failure_category", "erreur_lecture")]
    assert add_vote(client, "r1", 0, "c'est 41", "2026-10-03", llm=FakeJudge()) is None
    assert len(client.examples) == 1


def test_add_vote_waits_for_a_complete_trace():
    with pytest.raises(TraceNotReady):
        add_vote(FakeLangSmith({}), "r1", 1, None, "")
    with pytest.raises(TraceNotReady):
        add_vote(FakeLangSmith({"r1": FakeRun("r1", done=False)}), "r1", 1, None, "")


def test_background_collect_retries_until_the_trace_arrives():
    client = FakeLangSmith({})
    calls = []
    real_read = client.read_run

    def read_run(run_id):
        calls.append(run_id)
        if len(calls) == 2:
            client.runs["r1"] = FakeRun("r1")
        return real_read(run_id)

    client.read_run = read_run
    feedback_dataset._collect_with_retries("r1", 1, None, "", delays=(0, 0, 0), client=client, llm=FakeJudge())
    assert len(calls) == 2 and client.examples[0]["outputs"] == {"answer": "42 M€ [1]"}
