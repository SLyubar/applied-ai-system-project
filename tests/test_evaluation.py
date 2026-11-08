import pytest

from src.evaluation import (
    baseline_rank,
    check_parse,
    evaluate_parser,
    evaluate_ranking,
    format_report,
    is_deterministic,
    load_golden,
    main,
    precision_at_k,
    reciprocal_rank,
    summarize,
)
from src.guardrails import known_labels, sanitize_profile
from src.parser import parse_query
from src.recommender import load_songs


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture(scope="module")
def golden():
    return load_golden()


# ---- metrics ----

def test_precision_at_k_counts_hits_in_top_k():
    assert precision_at_k([1, 2, 3, 4, 5], [1, 3, 9], k=5) == pytest.approx(0.4)
    assert precision_at_k([1, 2, 3, 4, 5], [1, 3, 9], k=2) == pytest.approx(0.5)


def test_precision_at_k_uses_k_as_denominator_when_results_are_short():
    assert precision_at_k([1], [1], k=5) == pytest.approx(0.2)


def test_precision_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        precision_at_k([1], [1], k=0)


def test_reciprocal_rank():
    assert reciprocal_rank([7, 8, 9], [9]) == pytest.approx(1 / 3)
    assert reciprocal_rank([7, 8, 9], [7]) == 1.0
    assert reciprocal_rank([7, 8, 9], [42]) == 0.0
    assert reciprocal_rank([], [1]) == 0.0


def test_check_parse_exact_fields_and_bounds():
    profile = {"genre": "lofi", "mood": "chill", "energy": 0.35, "likes_acoustic": True}
    assert check_parse(profile, {"genre": "lofi", "energy_max": 0.4}) == []
    assert check_parse(profile, {"genre": "jazz"}) == ["genre"]
    assert check_parse(profile, {"energy_min": 0.5}) == ["energy_min"]
    assert check_parse(profile, {"energy_max": 0.3}) == ["energy_max"]


def test_check_parse_expected_false_does_not_match_missing_preference():
    assert check_parse({"energy": 0.5}, {"likes_acoustic": False}) == ["likes_acoustic"]


# ---- golden set integrity ----

def test_golden_relevant_ids_exist_in_catalog(golden, songs):
    ids = {s["id"] for s in songs}
    for case in golden["ranking_cases"]:
        assert set(case["relevant_ids"]) <= ids, case["name"]


def test_golden_cases_are_well_formed(golden):
    names = [c["name"] for c in golden["ranking_cases"]]
    assert len(names) == len(set(names))
    for case in golden["ranking_cases"]:
        assert case["relevant_ids"] or case.get("expect_warning"), (
            f"{case['name']} has no ground truth and no warning expectation, so it checks nothing"
        )
        if case.get("equivalent_to"):
            assert case["equivalent_to"] in names


def test_golden_parser_cases_use_known_fields(golden):
    allowed = {"genre", "mood", "likes_acoustic", "energy_min", "energy_max"}
    for case in golden["parser_cases"]:
        assert case["expected"], case["query"]
        assert set(case["expected"]) <= allowed, case["query"]
        assert case["tier"] in {"easy", "hard"}


def test_golden_expected_labels_exist_in_catalog(golden, songs):
    genres, moods = known_labels(songs)
    for case in golden["parser_cases"]:
        exp = case["expected"]
        assert exp.get("genre", genres[0]) in genres, case["query"]
        assert exp.get("mood", moods[0]) in moods, case["query"]


def test_golden_profiles_are_valid_inputs(golden):
    for case in golden["ranking_cases"]:
        sanitize_profile(case["profile"])  # must not raise


# ---- running the harness with fake pipelines ----

def fake_rank(ids, warnings=()):
    return lambda profile, songs, k: (list(ids)[:k], list(warnings))


def test_perfect_ranker_scores_perfectly(songs):
    cases = [{"name": "c", "profile": {}, "relevant_ids": [1, 2, 3, 4, 5]}]
    [result] = evaluate_ranking(cases, songs, fake_rank([1, 2, 3, 4, 5]))
    assert result["precision"] == 1.0
    assert result["rr"] == 1.0
    assert result["passed"]


def test_bad_ranker_fails_top1_check(songs):
    cases = [{"name": "c", "profile": {}, "relevant_ids": [1]}]
    [result] = evaluate_ranking(cases, songs, fake_rank([9, 8, 7, 6, 1]))
    assert result["top1_hit"] is False
    assert not result["passed"]


def test_expect_warning_fails_when_no_warning_is_raised(songs):
    cases = [{"name": "c", "profile": {}, "relevant_ids": [], "expect_warning": True}]
    [silent] = evaluate_ranking(cases, songs, fake_rank([1]))
    [warned] = evaluate_ranking(cases, songs, fake_rank([1], warnings=["careful"]))
    assert not silent["passed"]
    assert warned["passed"]


def test_equivalent_to_requires_identical_rankings(songs):
    cases = [
        {"name": "clean", "profile": {"v": 1}, "relevant_ids": [1]},
        {"name": "messy", "profile": {"v": 2}, "relevant_ids": [1], "equivalent_to": "clean"},
    ]

    def rank_fn(profile, songs, k):
        return ([1, 2, 3] if profile["v"] == 1 else [1, 3, 2]), []

    results = {r["name"]: r for r in evaluate_ranking(cases, songs, rank_fn)}
    assert results["clean"]["passed"]
    assert not results["messy"]["passed"]


def test_nondeterministic_pipeline_is_detected(songs):
    calls = {"n": 0}

    def flaky(profile, songs, k):
        calls["n"] += 1
        return [calls["n"]], []

    assert not is_deterministic([{"profile": {}}], songs, flaky)


def test_parser_errors_are_recorded_not_raised(songs):
    def broken(query, genres, moods):
        raise ValueError("boom")

    [result] = evaluate_parser([{"query": "x", "expected": {"genre": "pop"}}], songs, broken)
    assert not result["passed"]
    assert "boom" in result["failed_fields"][0]


def test_summarize_handles_all_failing_and_empty_inputs():
    summary = summarize([], [])
    assert summary["mean_precision"] == 0.0
    assert summary["parser_overall"] == 0.0


# ---- the real baseline ----

def test_baseline_passes_every_ranking_case(golden, songs):
    results = evaluate_ranking(golden["ranking_cases"], songs, baseline_rank)
    failing = [r["name"] for r in results if not r["passed"]]
    assert failing == []


def test_baseline_is_deterministic(golden, songs):
    assert is_deterministic(golden["ranking_cases"], songs, baseline_rank)


def test_keyword_parser_handles_all_easy_cases(golden, songs):
    easy = [c for c in golden["parser_cases"] if c["tier"] == "easy"]
    results = evaluate_parser(easy, songs, parse_query)
    failing = [r["query"] for r in results if not r["passed"]]
    assert failing == []


# ---- report ----

def test_report_contains_every_section(golden, songs):
    ranking = evaluate_ranking(golden["ranking_cases"], songs, baseline_rank)
    parsing = evaluate_parser(golden["parser_cases"], songs, parse_query)
    report = format_report(ranking, parsing, True, songs, label="test run")
    for heading in ("# Evaluation report (test run)", "## Summary", "## Ranking cases",
                    "## Parser cases", "## Needs human review"):
        assert heading in report
    assert "Mean precision@5" in report


def test_main_writes_report_file(tmp_path, capsys):
    out = tmp_path / "report.md"
    assert main(["--out", str(out), "--label", "ci"]) == 0
    assert "# Evaluation report (ci)" in out.read_text(encoding="utf-8")
    assert "Evaluation report" in capsys.readouterr().out
