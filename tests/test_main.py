"""Tests for the command line runner in src/main.py."""

import json

import pytest

from src.main import PROFILES, main, print_recommendations
from src.recommender import load_songs


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def test_default_run_prints_every_built_in_profile_and_logs_nothing(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    main(["--log-file", str(log)])
    out = capsys.readouterr().out
    for name in PROFILES:
        assert f"PROFILE: {name}" in out
    assert not log.exists()


def test_query_run_prints_results_and_logs_the_interaction(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    main(["--query", "calm music to study to", "--log-file", str(log)])
    out = capsys.readouterr().out
    assert "PROFILE: Query: calm music to study to" in out
    assert "Logged as " in out and "python -m src.feedback rate" in out
    [record] = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["query"] == "calm music to study to"
    assert record["parser"] == "keyword"
    assert len(record["recommendations"]) == 5
    assert record["profile"]["energy"] <= 0.45


def test_no_log_flag_skips_logging(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    main(["--query", "calm music", "--no-log", "--log-file", str(log)])
    assert "Logged as" not in capsys.readouterr().out
    assert not log.exists()


def test_results_show_scores_and_reasons(tmp_path, capsys):
    main(["--query", "angry metal", "--no-log"])
    out = capsys.readouterr().out
    assert "Score:" in out and "/ 1.00" in out and "Why:" in out
    assert "Iron Throne" in out


def test_print_recommendations_returns_the_cleaned_profile_and_results(songs, capsys):
    messy = {"genre": "Lofi ", "mood": "CHILL", "energy": 1.5, "likes_acoustic": True}
    profile, recs, _ = print_recommendations("messy", messy, songs, k=3)
    out = capsys.readouterr().out
    assert profile["genre"] == "lofi" and profile["energy"] == 1.0
    assert len(recs) == 3 and {"id", "title", "score"} == set(recs[0])
    assert "clamped" in out and "normalized" in out


def test_unrecognized_query_still_runs_and_explains_why(capsys):
    main(["--query", "asdkjh qwpoeiu", "--no-log"])
    out = capsys.readouterr().out
    assert "no genre recognized" in out and "no mood recognized" in out


def test_gemini_parser_without_a_key_falls_back_and_says_so(monkeypatch, capsys):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    main(["--query", "calm music to study to", "--parser", "gemini", "--no-log"])
    assert "Gemini parser unavailable (no API credentials)" in capsys.readouterr().out


def test_unknown_parser_is_rejected():
    with pytest.raises(SystemExit):
        main(["--query", "x", "--parser", "claude"])


# ---- confidence in the output ----

def run_profile(songs, profile, capsys):
    _, _, confidence = print_recommendations("p", profile, songs)
    return confidence, capsys.readouterr().out


def test_a_clear_profile_prints_high_confidence(songs, capsys):
    confidence, out = run_profile(songs, PROFILES["Starter lofi"], capsys)
    assert confidence.label == "high"
    assert "Confidence: HIGH" in out and "matched on: energy, mood, genre, acoustic" in out
    assert "Low confidence" not in out


def test_unknown_labels_print_low_confidence_with_reasons(songs, capsys):
    confidence, out = run_profile(songs, PROFILES["Unknown genre and mood"], capsys)
    assert confidence.label == "low"
    assert "Confidence: LOW" in out
    assert "genre 'vaporwave' is not in the catalog" in out
    assert "Low confidence: treat these as rough suggestions" in out


def test_a_contradictory_profile_is_not_high(songs, capsys):
    confidence, _ = run_profile(songs, PROFILES["Metal fan wants calm"], capsys)
    assert confidence.label != "high"


def test_confidence_is_logged_with_the_interaction(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    main(["--query", "asdkjh qwpoeiu zzzz", "--log-file", str(log)])
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["confidence"]["label"] == "low"
    assert record["confidence"]["reasons"]
    assert "Confidence: LOW" in capsys.readouterr().out


def test_gibberish_is_low_and_a_clear_request_is_high(capsys):
    main(["--query", "asdkjh qwpoeiu zzzz", "--no-log"])
    assert "Confidence: LOW" in capsys.readouterr().out
    main(["--query", "angry metal", "--no-log"])
    assert "Confidence: HIGH" in capsys.readouterr().out
