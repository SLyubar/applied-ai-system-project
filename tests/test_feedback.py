import json

import pytest

from src.feedback import (
    candidate_golden_cases,
    format_list,
    load_interactions,
    log_interaction,
    main,
    record_feedback,
)

RECS = [
    {"id": 4, "title": "Library Rain", "score": 0.9812},
    {"id": 2, "title": "Midnight Coding", "score": 0.9},
    {"id": 9, "title": "Focus Flow", "score": 0.7},
    {"id": 6, "title": "Spacewalk Thoughts", "score": 0.6},
]


@pytest.fixture
def log(tmp_path):
    return str(tmp_path / "logs" / "interactions.jsonl")  # the folder does not exist yet


def add(log, query="calm music to study to", **kwargs):
    return log_interaction(query, "keyword", {"genre": "lofi", "mood": "chill", "energy": 0.35}, kwargs.get("notes", []), RECS, log)


# ---- logging ----

def test_log_interaction_creates_the_folder_and_writes_one_json_line(log):
    interaction_id = add(log)
    lines = open(log, encoding="utf-8").read().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "interaction" and record["id"] == interaction_id
    assert len(interaction_id) == 8
    assert record["query"] == "calm music to study to"
    assert record["recommendations"][0] == {"id": 4, "title": "Library Rain", "score": 0.981}


def test_ids_are_unique(log):
    assert len({add(log) for _ in range(20)}) == 20


def test_non_english_queries_round_trip(log):
    add(log, query="雨の日に聴きたい落ち着いた曲")
    assert load_interactions(log)[0]["query"] == "雨の日に聴きたい落ち着いた曲"


def test_missing_log_loads_as_empty(log):
    assert load_interactions(log) == []


def test_torn_or_blank_lines_do_not_break_the_log(log):
    add(log)
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n{not json\n")
    add(log, query="second")
    assert [r["query"] for r in load_interactions(log)] == ["calm music to study to", "second"]


# ---- feedback ----

def test_rating_is_attached_to_the_interaction(log):
    interaction_id = add(log)
    record_feedback(interaction_id, "bad", "too loud", log)
    [r] = load_interactions(log)
    assert (r["rating"], r["note"]) == ("bad", "too loud")


def test_latest_rating_wins(log):
    interaction_id = add(log)
    record_feedback(interaction_id, "bad", path=log)
    record_feedback(interaction_id, "good", path=log)
    assert load_interactions(log)[0]["rating"] == "good"


def test_unrated_interactions_have_no_rating(log):
    add(log)
    assert load_interactions(log)[0]["rating"] is None


def test_rating_an_unknown_id_is_rejected(log):
    add(log)
    with pytest.raises(ValueError):
        record_feedback("deadbeef", "good", path=log)


def test_invalid_rating_is_rejected(log):
    interaction_id = add(log)
    with pytest.raises(ValueError):
        record_feedback(interaction_id, "meh", path=log)


def test_feedback_does_not_overwrite_the_original_record(log):
    interaction_id = add(log)
    record_feedback(interaction_id, "good", path=log)
    types = [json.loads(line)["type"] for line in open(log, encoding="utf-8")]
    assert types == ["interaction", "feedback"]


# ---- candidate golden cases ----

def test_only_bad_queries_become_candidates(log):
    good, bad, unrated = add(log, query="a"), add(log, query="b"), add(log, query="c")
    record_feedback(good, "good", path=log)
    record_feedback(bad, "bad", "wrong mood", log)
    cases = candidate_golden_cases(log)
    assert [c["query"] for c in cases] == ["b"]
    assert unrated not in str(cases)


def test_candidates_leave_the_judgments_to_a_human(log):
    interaction_id = add(log, query="x" * 100)
    record_feedback(interaction_id, "bad", "wrong mood", log)
    [case] = candidate_golden_cases(log)
    assert case["ideal_ids"] == [] and case["acceptable_ids"] == []
    assert case["forbidden_ids"] == [4, 2, 9]  # the top 3 that were shown
    assert "wrong mood" in case["notes"] and "Library Rain" in case["notes"]
    assert len(case["name"]) < 70


def test_no_bad_ratings_means_no_candidates(log):
    interaction_id = add(log)
    record_feedback(interaction_id, "good", path=log)
    assert candidate_golden_cases(log) == []


# ---- CLI ----

def test_cli_list_on_an_empty_log(log, capsys):
    assert main(["--log-file", log, "list"]) == 0
    assert "No interactions logged yet" in capsys.readouterr().out


def test_cli_list_shows_rating_and_top_result(log, capsys):
    interaction_id = add(log)
    record_feedback(interaction_id, "good", path=log)
    main(["--log-file", log, "list"])
    out = capsys.readouterr().out
    assert interaction_id in out and "[good]" in out and "Library Rain" in out


def test_cli_rate_success_and_unknown_id(log, capsys):
    interaction_id = add(log)
    assert main(["--log-file", log, "rate", interaction_id, "bad", "--note", "nope"]) == 0
    assert main(["--log-file", log, "rate", "deadbeef", "good"]) == 1
    captured = capsys.readouterr()
    assert "Recorded 'bad'" in captured.out and "no logged interaction" in captured.err


def test_cli_export_candidates_prints_valid_json(log, capsys):
    record_feedback(add(log, query="why"), "bad", path=log)
    main(["--log-file", log, "export-candidates"])
    exported = json.loads(capsys.readouterr().out)
    assert exported[0]["query"] == "why"


def test_format_list_handles_empty():
    assert "No interactions" in format_list([])


def test_confidence_is_stored_and_shown_in_the_list(log, capsys):
    interaction_id = log_interaction("q", "keyword", {}, [], RECS, log, confidence={"label": "low", "score": 0.2})
    assert load_interactions(log)[0]["confidence"]["label"] == "low"
    main(["--log-file", log, "list"])
    assert interaction_id in capsys.readouterr().out
    assert "(low confidence)" in format_list(load_interactions(log))


def test_old_log_entries_without_confidence_still_list(log):
    log_interaction("q", "keyword", {}, [], RECS, log)  # confidence omitted
    assert "confidence" not in format_list(load_interactions(log))
