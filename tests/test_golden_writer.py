"""Tests for src/golden_writer.py. Every test works on a temp copy of the golden set."""

import json
import shutil

import pytest

from src.evaluation import GOLDEN_PATH, case_grades
from src.golden_writer import find_ui_case, save_case
from src.recommender import load_songs


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture
def golden(tmp_path):
    path = tmp_path / "golden_set.json"
    shutil.copyfile(GOLDEN_PATH, path)
    return str(path)


def cases(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["ranking_cases"]


def read_raw(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def test_save_adds_a_ui_case_in_the_golden_schema(golden, songs):
    before = len(cases(golden))
    case, action = save_case("sleepy piano", [4, 2], [21], [17], songs, "my note", golden)
    assert action == "added"
    after = cases(golden)
    assert len(after) == before + 1 and after[-1] == case
    assert case["query"] == "sleepy piano" and case["source"] == "ui" and case["notes"] == "my note"
    assert case["ideal_ids"] == [4, 2] and case["acceptable_ids"] == [21] and case["forbidden_ids"] == [17]
    assert case_grades(case) == {4: 2, 2: 2, 21: 1}  # the evaluation reads it like any other case


def test_existing_content_and_file_format_are_untouched(golden, songs):
    raw_before = read_raw(golden)
    save_case("sleepy piano", [4], [], [], songs, path=golden)
    raw_after = read_raw(golden)
    old, new = json.loads(raw_before), json.loads(raw_after)
    assert new["ranking_cases"][:-1] == old["ranking_cases"]
    assert {k: v for k, v in new.items() if k != "ranking_cases"} == {k: v for k, v in old.items() if k != "ranking_cases"}
    assert ("\r\n" in raw_after) == ("\r\n" in raw_before)
    assert raw_after.endswith("\n") == raw_before.endswith("\n")
    assert raw_after.startswith(raw_before[:200])


def test_saving_again_updates_the_same_case(golden, songs):
    save_case("Sleepy Piano", [4], [], [], songs, path=golden)
    count = len(cases(golden))
    case, action = save_case("sleepy piano ", [2], [21], [17], songs, path=golden)
    assert action == "updated" and len(cases(golden)) == count
    assert case["ideal_ids"] == [2] and case["name"] == "UI: Sleepy Piano"


def test_hand_curated_case_for_the_same_query_is_never_overwritten(golden, songs):
    curated = next(c for c in cases(golden) if c.get("query"))
    before = cases(golden)
    with pytest.raises(ValueError, match="hand-curated"):
        save_case(curated["query"].upper(), [4], [], [], songs, path=golden)
    assert cases(golden) == before


def test_duplicate_names_get_a_suffix(golden, songs):
    save_case("x" * 40 + "aaa", [4], [], [], songs, path=golden)
    case, _ = save_case("x" * 40 + "bbb", [4], [], [], songs, path=golden)
    assert case["name"].endswith("(2)")
    names = [c["name"] for c in cases(golden)]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "args, message",
    [
        (("  ", [4], [], []), "query is empty"),
        (("q", [], [], [17]), "at least one"),
        (("q", [9999], [], []), "unknown song ids in ideal"),
        (("q", [4], [], [9999]), "unknown song ids in forbidden"),
        (("q", [4], [4], []), "two grades"),
        (("q", [4], [], [4]), "two grades"),
    ],
)
def test_invalid_input_is_rejected_and_file_unchanged(golden, songs, args, message):
    raw = read_raw(golden)
    with pytest.raises(ValueError, match=message):
        save_case(*args, songs, path=golden)
    assert read_raw(golden) == raw


def test_duplicate_ids_in_one_list_are_collapsed(golden, songs):
    case, _ = save_case("q", [4, 4, 2], [], [], songs, path=golden)
    assert case["ideal_ids"] == [4, 2]


def test_blank_note_gets_a_default(golden, songs):
    case, _ = save_case("q", [4], [], [], songs, "  ", golden)
    assert "Streamlit" in case["notes"]


def test_line_endings_and_trailing_newline_are_preserved(tmp_path, songs):
    path = tmp_path / "g.json"
    body = json.dumps({"ranking_cases": []}, indent=2)
    path.write_bytes((body + "\n").encode("utf-8"))  # LF with trailing newline
    save_case("q", [4], [], [], songs, path=str(path))
    raw = path.read_bytes().decode("utf-8")
    assert "\r" not in raw and raw.endswith("\n")
    path.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))  # CRLF, no trailing newline
    save_case("q", [4], [], [], songs, path=str(path))
    raw = path.read_bytes().decode("utf-8")
    assert "\r\n" in raw and "\n" not in raw.replace("\r\n", "") and not raw.endswith("\n")


def test_failed_write_leaves_no_temp_file_and_original_intact(golden, songs, monkeypatch, tmp_path):
    import src.golden_writer as gw

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(gw.os, "replace", boom)
    raw = read_raw(golden)
    with pytest.raises(OSError):
        save_case("q", [4], [], [], songs, path=golden)
    assert read_raw(golden) == raw
    assert not list(tmp_path.glob("*.tmp"))


def test_find_ui_case_only_returns_ui_cases(golden, songs):
    curated = next(c for c in cases(golden) if c.get("query"))
    assert find_ui_case(curated["query"], golden) == {}
    assert find_ui_case("nothing here", golden) == {}
    case, _ = save_case("Sleepy Piano", [4], [], [], songs, path=golden)
    assert find_ui_case(" sleepy piano", golden) == case


def test_find_ui_case_tolerates_a_missing_or_broken_file(tmp_path):
    assert find_ui_case("q", str(tmp_path / "missing.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert find_ui_case("q", str(bad)) == {}


def test_saved_set_still_passes_the_golden_integrity_rules(golden, songs):
    save_case("sleepy piano", [4, 2], [21], [17], songs, path=golden)
    ranking = cases(golden)
    names = [c["name"] for c in ranking]
    assert len(names) == len(set(names))
    for c in ranking:
        assert ("profile" in c) != ("query" in c)
        assert c.get("ideal_ids") or c.get("acceptable_ids") or c.get("expect_warning")
