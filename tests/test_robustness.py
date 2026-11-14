import pytest

from src.evaluation import FALLBACK_NOTE_PREFIX, check_parse, load_golden, main
from src.guardrails import known_labels
from src.recommender import load_songs
from src.robustness import (
    MAX_ENERGY_SPREAD,
    check_safe,
    evaluate_adversarial,
    evaluate_paraphrases,
    format_section,
    summarize_robustness,
)

GENRES = ["folk", "lofi"]
MOODS = ["chill", "happy"]


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture(scope="module")
def golden():
    return load_golden()


def constant_parser(profile, notes=()):
    return lambda query, genres, moods: (dict(profile), list(notes))


# ---- check_parse set membership ----

def test_check_parse_supports_allowed_sets():
    profile = {"genre": "lofi", "mood": "chill", "energy": 0.3}
    assert check_parse(profile, {"mood_in": ["chill", "relaxed"]}) == []
    assert check_parse(profile, {"mood_in": ["happy"]}) == ["mood_in"]
    assert check_parse(profile, {"genre_in": ["folk"]}) == ["genre_in"]


# ---- safety check ----

def test_valid_profile_is_safe():
    assert check_safe({"genre": "lofi", "mood": "", "energy": 0.5, "likes_acoustic": True}, GENRES, MOODS) == []


@pytest.mark.parametrize("profile", [
    {"genre": "hacked", "mood": "chill", "energy": 0.5},
    {"genre": "lofi", "mood": "pwned", "energy": 0.5},
    {"genre": "lofi", "mood": "chill", "energy": 99},
    {"genre": "lofi", "mood": "chill", "energy": -0.1},
    {"genre": "lofi", "mood": "chill", "energy": None},
    {"genre": "lofi", "mood": "chill", "energy": True},
    {"genre": "lofi", "mood": "chill", "energy": 0.5, "likes_acoustic": "yes"},
])
def test_unsafe_profiles_are_rejected(profile):
    assert check_safe(profile, GENRES, MOODS)


# ---- paraphrase groups ----

GROUP = {"name": "g", "expected": {"energy_max": 0.45}, "queries": ["a", "b", "c"]}


def test_consistent_paraphrases_pass(songs):
    parser = constant_parser({"genre": "", "mood": "", "energy": 0.3})
    [result] = evaluate_paraphrases([GROUP], songs, parser)
    assert result["passed"] and result["phrases_passed"] == 3 and result["energy_spread"] == 0


def test_a_wording_that_misses_the_expectation_fails_the_group(songs):
    energies = iter([0.3, 0.3, 0.9])
    parser = lambda q, g, m: ({"genre": "", "mood": "", "energy": next(energies)}, [])
    [result] = evaluate_paraphrases([GROUP], songs, parser)
    assert not result["passed"]
    assert result["phrases_passed"] == 2
    assert "'c'" in result["failures"][0]


def test_wordings_that_each_pass_but_disagree_fail_on_spread(songs):
    group = {"name": "g", "expected": {"energy_max": 0.45}, "queries": ["a", "b"]}
    energies = iter([0.05, 0.40])  # both under the bound, but 0.35 apart
    parser = lambda q, g, m: ({"genre": "", "mood": "", "energy": next(energies)}, [])
    [result] = evaluate_paraphrases([group], songs, parser)
    assert result["phrases_passed"] == 2
    assert result["energy_spread"] == pytest.approx(0.35) and result["energy_spread"] > MAX_ENERGY_SPREAD
    assert not result["passed"]


def test_paraphrase_parse_errors_are_recorded(songs):
    def broken(query, genres, moods):
        raise ValueError("empty")

    [result] = evaluate_paraphrases([GROUP], songs, broken)
    assert not result["passed"] and result["phrases_passed"] == 0


def test_paraphrase_fallbacks_are_counted(songs):
    parser = constant_parser({"genre": "", "mood": "", "energy": 0.3}, [FALLBACK_NOTE_PREFIX + " (x)"])
    [result] = evaluate_paraphrases([GROUP], songs, parser)
    assert result["fallbacks"] == 3


# ---- adversarial cases ----

def test_safe_profile_meeting_expectations_passes(songs):
    case = {"name": "c", "query": "x", "expected": {"genre": ""}}
    [result] = evaluate_adversarial([case], songs, constant_parser({"genre": "", "mood": "", "energy": 0.5}))
    assert result["passed"]


def test_unsafe_profile_fails_even_with_no_expectations(songs):
    case = {"name": "c", "query": "x", "expected": {}}
    [result] = evaluate_adversarial([case], songs, constant_parser({"genre": "hacked", "mood": "", "energy": 0.5}))
    assert not result["passed"] and result["problems"]


def test_unmet_expectation_fails(songs):
    case = {"name": "c", "query": "x", "expected": {"energy_max": 0.45}}
    [result] = evaluate_adversarial([case], songs, constant_parser({"genre": "", "mood": "", "energy": 0.5}))
    assert not result["passed"] and result["failed_fields"] == ["energy_max"]


def test_query_repeat_builds_a_long_query(songs):
    seen = {}

    def spy(query, genres, moods):
        seen["query"] = query
        return {"genre": "", "mood": "", "energy": 0.5}, []

    evaluate_adversarial([{"name": "long", "query": "ab ", "query_repeat": 100}], songs, spy)
    assert seen["query"] == "ab " * 100


def test_adversarial_parse_errors_are_failures_not_crashes(songs):
    def broken(query, genres, moods):
        raise ValueError("empty")

    [result] = evaluate_adversarial([{"name": "c", "query": "", "expected": {}}], songs, broken)
    assert not result["passed"]


# ---- reporting ----

def make_run(songs, parser):
    return {
        "paraphrase": evaluate_paraphrases([GROUP], songs, parser),
        "adversarial": evaluate_adversarial([{"name": "adv", "query": "x", "expected": {}}], songs, parser),
    }


def test_section_shows_pass_counts_across_runs(songs):
    good = constant_parser({"genre": "", "mood": "", "energy": 0.3})
    bad = constant_parser({"genre": "", "mood": "", "energy": 0.9})
    report = format_section([make_run(songs, good), make_run(songs, bad), make_run(songs, good)])
    assert "Runs: 3" in report
    assert "| g | 2/3 |" in report
    assert "| adv | 3/3 |" in report
    assert "did not pass every run: g." in report


def test_section_says_so_when_everything_passed(songs):
    report = format_section([make_run(songs, constant_parser({"genre": "", "mood": "", "energy": 0.3}))])
    assert "every case passed every run" in report


def test_section_warns_about_fallbacks(songs):
    parser = constant_parser({"genre": "", "mood": "", "energy": 0.3}, [FALLBACK_NOTE_PREFIX + " (x)"])
    run = make_run(songs, parser)
    assert summarize_robustness(run["paraphrase"], run["adversarial"])["fallbacks"] == 4
    assert "came from the keyword fallback" in format_section([run])


# ---- golden set integrity for the new sections ----

def test_golden_has_enough_robustness_cases(golden):
    assert len(golden["paraphrase_groups"]) >= 5
    assert all(len(g["queries"]) >= 3 for g in golden["paraphrase_groups"])
    names = " ".join(c["name"] for c in golden["adversarial_cases"])
    for needed in ("Injection", "Spanish", "Gibberish", "Very long"):
        assert needed in names


def test_golden_robustness_labels_exist_in_catalog(golden, songs):
    genres, moods = known_labels(songs)
    expectations = [g["expected"] for g in golden["paraphrase_groups"]] + [
        c["expected"] for c in golden["adversarial_cases"]
    ]
    for exp in expectations:
        assert exp.get("genre", "") in genres + [""]
        assert exp.get("mood", "") in moods + [""]
        assert set(exp.get("mood_in", moods)) <= set(moods)


def test_calm_and_energetic_expectations_cannot_be_met_by_the_default_energy(golden):
    """A parser that understood nothing returns energy 0.5. That must never satisfy a mood expectation."""
    # Injection cases are the exception: their bound means "did not obey the injected 99",
    # and for them the default energy of 0.5 is exactly the right answer.
    expectations = [g["expected"] for g in golden["paraphrase_groups"]] + [
        c["expected"] for c in golden["adversarial_cases"] if not c["name"].startswith("Injection")
    ]
    for exp in expectations:
        assert exp.get("energy_max", 0.0) < 0.5
        assert exp.get("energy_min", 1.0) > 0.5


# ---- end to end through the CLI ----

def test_cli_report_includes_the_robustness_section(tmp_path):
    out = tmp_path / "r.md"
    main(["--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "## Robustness" in text and "### Paraphrase groups" in text


def test_strict_mode_fails_when_robustness_cases_fail():
    # The keyword baseline cannot parse non-English requests, so strict mode must report failure.
    assert main(["--strict"]) == 1


# ---- understanding expectations ----

def test_junk_that_is_rated_as_understood_fails_the_case(songs):
    case = {"name": "junk", "query": "x", "expected": {"understanding_max": 0.5}}
    confident = constant_parser({"genre": "lofi", "mood": "chill", "energy": 0.4})
    empty = constant_parser({"genre": "", "mood": "", "energy": 0.5}, ["no energy cue found; defaulted to 0.5"])
    [bad] = evaluate_adversarial([case], songs, confident)
    [good] = evaluate_adversarial([case], songs, empty)
    assert not bad["passed"] and "understanding_max" in bad["failed_fields"]
    assert good["passed"] and good["understanding"] == 0.0


def test_a_clear_request_rated_as_confusing_fails_the_case(songs):
    case = {"name": "clear", "query": "x", "expected": {"understanding_min": 0.6}}
    doubtful = constant_parser({"genre": "lofi", "mood": "chill", "energy": 0.4, "parser_confidence": 0.3})
    sure = constant_parser({"genre": "lofi", "mood": "chill", "energy": 0.4, "parser_confidence": 0.9})
    assert not evaluate_adversarial([case], songs, doubtful)[0]["passed"]
    assert evaluate_adversarial([case], songs, sure)[0]["passed"]


def test_junk_cases_in_the_golden_set_expect_low_understanding(golden):
    by_name = {c["name"]: c["expected"] for c in golden["adversarial_cases"]}
    for name in ("Gibberish", "Off-topic question", "Punctuation only", "Injection: reply with a word"):
        assert by_name[name]["understanding_max"] <= 0.5
    for name in ("Spanish", "Japanese", "French", "German"):
        assert by_name[name]["understanding_min"] >= 0.6


def test_the_keyword_parser_rates_junk_as_not_understood(golden, songs):
    junk = [c for c in golden["adversarial_cases"] if "understanding_max" in c["expected"]]
    from src.parser import parse_query

    results = evaluate_adversarial(junk, songs, parse_query)
    assert all(r["passed"] for r in results), [r["name"] for r in results if not r["passed"]]
