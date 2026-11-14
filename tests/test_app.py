"""End-to-end tests of the Streamlit app (app.py) using Streamlit's headless AppTest.

The app is pointed at temp copies of the golden set and the log, so the real files are never
touched, and only the offline keyword parser is used, so no API calls are made.
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.evaluation import GOLDEN_PATH  # noqa: E402
from src.feedback import load_interactions  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def files(tmp_path, monkeypatch):
    golden, log = tmp_path / "golden_set.json", tmp_path / "log.jsonl"
    shutil.copyfile(GOLDEN_PATH, golden)
    monkeypatch.setenv("RECOMMENDER_GOLDEN_FILE", str(golden))
    monkeypatch.setenv("RECOMMENDER_LOG_FILE", str(log))
    return golden, log


def golden_cases(path):
    return json.loads(path.read_text(encoding="utf-8"))["ranking_cases"]


def button(at, label):
    return next(b for b in at.button if b.label == label)


def search(query="calm music to study to"):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input(key="query").set_value(query)
    button(at, "Recommend").click().run()
    return at


def test_page_loads_with_no_results_and_no_errors(files):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert not at.radio  # nothing to judge until a search is run


def test_search_shows_recommendations_and_logs_the_interaction(files):
    _, log = files
    at = search()
    assert not at.exception
    assert len(at.radio) == 5 + 1  # one judgment per song plus the overall rating
    assert any("Confidence" in m.value for m in at.markdown)
    [interaction] = load_interactions(str(log))
    assert interaction["query"] == "calm music to study to" and interaction["rating"] is None


def test_empty_query_shows_an_error_and_no_results(files):
    at = search("   ")
    assert [e.value for e in at.error] == ["Type a request first."]
    assert not at.radio


def test_saving_judgments_writes_a_case_to_the_golden_set_and_the_rating_to_the_log(files):
    golden, log = files
    before = golden_cases(golden)
    at = search()
    songs = [r for r in at.radio if r.key and r.key.startswith("j-")]
    songs[0].set_value("Ideal")
    songs[1].set_value("Fine")
    songs[4].set_value("Must not appear")
    next(r for r in at.radio if r.label.startswith("Overall")).set_value("Bad")
    button(at, "Save feedback").click().run()
    assert not at.exception and not at.error

    after = golden_cases(golden)
    assert after[:-1] == before and len(after) == len(before) + 1
    case = after[-1]
    ids = [int(r.key.rsplit("-", 1)[1]) for r in songs]
    assert case["query"] == "calm music to study to" and case["source"] == "ui"
    assert case["ideal_ids"] == [ids[0]] and case["acceptable_ids"] == [ids[1]] and case["forbidden_ids"] == [ids[4]]
    assert load_interactions(str(log))[0]["rating"] == "bad"


def test_saving_the_same_query_again_updates_instead_of_duplicating(files):
    golden, _ = files
    for choice in ("Ideal", "Fine"):
        at = search()
        next(r for r in at.radio if r.key and r.key.startswith("j-")).set_value(choice)
        button(at, "Save feedback").click().run()
    cases = [c for c in golden_cases(golden) if c.get("source") == "ui"]
    assert len(cases) == 1 and cases[0]["ideal_ids"] == [] and len(cases[0]["acceptable_ids"]) == 1


def test_only_forbidden_songs_is_refused_with_a_clear_message_and_golden_set_unchanged(files):
    golden, _ = files
    before = golden.read_text(encoding="utf-8")
    at = search()
    next(r for r in at.radio if r.key and r.key.startswith("j-")).set_value("Must not appear")
    button(at, "Save feedback").click().run()
    assert any("Golden set not changed" in e.value and "at least one" in e.value for e in at.error)
    assert golden.read_text(encoding="utf-8") == before


def test_overall_rating_alone_is_logged_without_touching_the_golden_set(files):
    golden, log = files
    before = golden.read_text(encoding="utf-8")
    at = search()
    next(r for r in at.radio if r.label.startswith("Overall")).set_value("Good")
    button(at, "Save feedback").click().run()
    assert not at.error
    assert load_interactions(str(log))[0]["rating"] == "good"
    assert golden.read_text(encoding="utf-8") == before


def test_nothing_selected_changes_nothing(files):
    golden, log = files
    before = golden.read_text(encoding="utf-8")
    at = search()
    button(at, "Save feedback").click().run()
    assert not at.error and golden.read_text(encoding="utf-8") == before
    assert load_interactions(str(log))[0]["rating"] is None


def test_a_hand_curated_query_cannot_be_overwritten_from_the_ui(files):
    golden, _ = files
    curated = next(c for c in golden_cases(golden) if c.get("query"))
    before = golden.read_text(encoding="utf-8")
    at = search(curated["query"])
    next(r for r in at.radio if r.key and r.key.startswith("j-")).set_value("Ideal")
    button(at, "Save feedback").click().run()
    assert any("hand-curated" in e.value for e in at.error)
    assert golden.read_text(encoding="utf-8") == before


def test_repeat_query_prefills_the_saved_judgments(files):
    at = search()
    first = next(r for r in at.radio if r.key and r.key.startswith("j-"))
    first.set_value("Ideal")
    button(at, "Save feedback").click().run()
    again = search()
    assert any("already judged" in i.value for i in again.info)
    prefilled = next(r for r in again.radio if r.key and r.key.startswith("j-"))
    assert prefilled.value == "Ideal"


def test_example_buttons_fill_the_query(files):
    at = AppTest.from_file(APP, default_timeout=30).run()
    example = at.button[0]
    example.click().run()
    assert at.text_input(key="query").value == example.label
