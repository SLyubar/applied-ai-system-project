import pytest

from src.evaluation import (
    MAX_RUNS,
    aggregate_runs,
    format_multi_run_report,
    FALLBACK_NOTE_PREFIX,
    baseline_rank,
    check_parse,
    evaluate_parser,
    evaluate_ranking,
    format_report,
    forbidden_in_top_k,
    is_deterministic,
    load_golden,
    main,
    ndcg_at_k,
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

def test_golden_song_ids_exist_in_catalog(golden, songs):
    ids = {s["id"] for s in songs}
    for case in golden["ranking_cases"]:
        for key in ("ideal_ids", "acceptable_ids", "forbidden_ids"):
            assert set(case.get(key, [])) <= ids, f"{case['name']}: {key}"


def test_golden_grades_do_not_contradict_each_other(golden):
    for case in golden["ranking_cases"]:
        ideal = set(case.get("ideal_ids", []))
        acceptable = set(case.get("acceptable_ids", []))
        forbidden = set(case.get("forbidden_ids", []))
        assert not ideal & acceptable, f"{case['name']}: a song is both ideal and acceptable"
        assert not (ideal | acceptable) & forbidden, f"{case['name']}: a song is both wanted and forbidden"


def test_golden_cases_are_well_formed(golden):
    names = [c["name"] for c in golden["ranking_cases"]]
    assert len(names) == len(set(names))
    for case in golden["ranking_cases"]:
        assert ("profile" in case) != ("query" in case), f"{case['name']} needs exactly one of profile/query"
        has_truth = case.get("ideal_ids") or case.get("acceptable_ids")
        assert has_truth or case.get("expect_warning"), (
            f"{case['name']} has no ground truth and no warning expectation, so it checks nothing"
        )
        if case.get("equivalent_to"):
            assert case["equivalent_to"] in names


def test_golden_set_is_large_enough_to_discriminate(golden):
    graded = [c for c in golden["ranking_cases"] if c.get("ideal_ids") or c.get("acceptable_ids")]
    assert len(graded) >= 15
    assert sum(1 for c in golden["ranking_cases"] if "query" in c) >= 5
    assert sum(1 for c in graded if c.get("forbidden_ids")) >= 12


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
        if "profile" in case:
            sanitize_profile(case["profile"])  # must not raise


# ---- running the harness with fake pipelines ----

def fake_rank(ids, warnings=()):
    return lambda profile, songs, k: (list(ids)[:k], list(warnings))


def test_perfect_ranker_scores_perfectly(songs):
    cases = [{"name": "c", "profile": {}, "ideal_ids": [1, 2], "acceptable_ids": [3, 4, 5]}]
    [result] = evaluate_ranking(cases, songs, fake_rank([1, 2, 3, 4, 5]))
    assert result["precision"] == 1.0
    assert result["rr"] == 1.0
    assert result["ndcg"] == pytest.approx(1.0)
    assert result["passed"]


def test_bad_ranker_fails_top1_check(songs):
    cases = [{"name": "c", "profile": {}, "acceptable_ids": [1]}]
    [result] = evaluate_ranking(cases, songs, fake_rank([9, 8, 7, 6, 1]))
    assert result["top1_hit"] is False
    assert not result["passed"]


def test_expect_warning_fails_when_no_warning_is_raised(songs):
    cases = [{"name": "c", "profile": {}, "expect_warning": True}]
    [silent] = evaluate_ranking(cases, songs, fake_rank([1]))
    [warned] = evaluate_ranking(cases, songs, fake_rank([1], warnings=["careful"]))
    assert not silent["passed"]
    assert warned["passed"]


def test_equivalent_to_requires_identical_rankings(songs):
    cases = [
        {"name": "clean", "profile": {"v": 1}, "acceptable_ids": [1]},
        {"name": "messy", "profile": {"v": 2}, "acceptable_ids": [1], "equivalent_to": "clean"},
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

def test_baseline_passes_every_profile_case(golden, songs):
    results = evaluate_ranking(golden["ranking_cases"], songs, baseline_rank)
    failing = [r["name"] for r in results if not r["is_query"] and not r["passed"]]
    assert failing == []


def test_baseline_never_recommends_a_forbidden_song(golden, songs):
    results = evaluate_ranking(golden["ranking_cases"], songs, baseline_rank)
    assert [(r["name"], r["forbidden_hits"]) for r in results if r["forbidden_hits"]] == []


def test_baseline_keyword_parser_fails_the_hard_queries_end_to_end(golden, songs):
    """Documents the headroom the Gemini parser is meant to close. If this starts
    failing because the keyword parser improved, update the baseline report."""
    results = {r["name"]: r for r in evaluate_ranking(golden["ranking_cases"], songs, baseline_rank)}
    assert not results["Query: fall asleep"]["passed"]
    assert not results["Query: warm hug"]["passed"]


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


# ---- graded relevance, forbidden songs, and query cases ----

def test_ndcg_is_one_for_the_best_possible_ranking():
    grades = {1: 2, 2: 1, 3: 1}
    assert ndcg_at_k([1, 2, 3, 9, 8], grades, k=5) == pytest.approx(1.0)


def test_ndcg_rewards_putting_the_ideal_song_first():
    grades = {1: 2, 2: 1}
    ideal_first = ndcg_at_k([1, 2, 9], grades, k=3)
    ideal_last = ndcg_at_k([9, 2, 1], grades, k=3)
    assert ideal_first > ideal_last > 0


def test_ndcg_is_zero_with_no_relevant_results_or_no_grades():
    assert ndcg_at_k([7, 8, 9], {1: 2}, k=3) == 0.0
    assert ndcg_at_k([1, 2], {}, k=3) == 0.0


def test_ndcg_rejects_non_positive_k():
    with pytest.raises(ValueError):
        ndcg_at_k([1], {1: 1}, k=0)


def test_forbidden_in_top_k_only_looks_at_the_top_k():
    assert forbidden_in_top_k([1, 2, 3, 4, 5, 6], [2, 6], k=5) == [2]
    assert forbidden_in_top_k([1, 2], [], k=5) == []


def test_forbidden_song_fails_a_case_even_if_top1_is_good(songs):
    cases = [{"name": "c", "profile": {}, "ideal_ids": [1], "forbidden_ids": [9]}]
    [result] = evaluate_ranking(cases, songs, fake_rank([1, 2, 9]))
    assert result["top1_hit"]
    assert result["forbidden_hits"] == [9]
    assert not result["passed"]


def test_ideal_song_must_appear_in_top_three(songs):
    cases = [{"name": "c", "profile": {}, "ideal_ids": [1], "acceptable_ids": [2]}]
    [late] = evaluate_ranking(cases, songs, fake_rank([2, 3, 4, 1]))
    [early] = evaluate_ranking(cases, songs, fake_rank([2, 1, 3, 4]))
    assert late["top1_hit"] and not late["passed"]
    assert early["passed"]


def test_ideal_check_is_skipped_when_a_case_has_no_ideal_songs(songs):
    cases = [{"name": "c", "profile": {}, "acceptable_ids": [2]}]
    [result] = evaluate_ranking(cases, songs, fake_rank([2, 3]))
    assert result["ideal_in_top3"] is None
    assert result["passed"]


def test_query_cases_are_parsed_before_ranking(songs):
    seen = {}

    def spy_rank(profile, songs, k):
        seen["profile"] = profile
        return [1], []

    def fake_parse(query, genres, moods):
        return {"genre": "lofi", "mood": "chill", "energy": 0.3}, []

    cases = [{"name": "q", "query": "something calm", "acceptable_ids": [1]}]
    [result] = evaluate_ranking(cases, songs, spy_rank, parse_fn=fake_parse)
    assert seen["profile"]["genre"] == "lofi"
    assert result["is_query"] and result["passed"]


def test_unparseable_query_fails_the_case_instead_of_crashing(songs):
    def broken(query, genres, moods):
        raise ValueError("empty")

    cases = [{"name": "q", "query": "", "acceptable_ids": [1]}]
    [result] = evaluate_ranking(cases, songs, fake_rank([1]), parse_fn=broken)
    assert not result["passed"]
    assert result["ranked"] == []


def test_fallback_in_a_query_case_is_flagged_and_counted(songs):
    def fell_back(query, genres, moods):
        return {"genre": "lofi", "mood": "chill", "energy": 0.3}, [FALLBACK_NOTE_PREFIX + " (test)"]

    cases = [{"name": "q", "query": "x", "acceptable_ids": [1]}]
    results = evaluate_ranking(cases, songs, fake_rank([1]), parse_fn=fell_back)
    assert results[0]["fell_back"]
    summary = summarize(results, [])
    assert summary["ranking_fallbacks"] == 1
    report = format_report(results, [], True, songs, label="gemini parser")
    assert "came from the keyword fallback" in report


def test_determinism_check_ignores_query_cases(songs):
    calls = {"n": 0}

    def flaky(profile, songs, k):
        calls["n"] += 1
        return [calls["n"]], []

    cases = [{"name": "q", "query": "x", "acceptable_ids": [1]}]
    assert is_deterministic(cases, songs, flaky)
    assert calls["n"] == 0


# ---- repeated runs (--runs) ----

def flaky_parser():
    """Passes on even calls, fails on odd calls, so a case is stable only by luck."""
    calls = {"n": 0}

    def parse(query, genres, moods):
        calls["n"] += 1
        genre = "folk" if calls["n"] % 2 == 0 else "lofi"
        return {"genre": genre, "mood": "", "energy": 0.5}, []

    return parse


def run_parser_n_times(songs, parse_fn, n):
    cases = [{"query": "relaxing acoustic folk", "tier": "easy", "expected": {"genre": "folk"}}]
    return [([], evaluate_parser(cases, songs, parse_fn)) for _ in range(n)]


def test_aggregate_runs_counts_passes_per_case(songs):
    agg = aggregate_runs(run_parser_n_times(songs, flaky_parser(), 4))
    [case] = agg["cases"]
    assert (case["passes"], case["runs"], case["kind"]) == (2, 4, "parser")
    assert len(agg["summaries"]) == 4


def test_aggregate_runs_includes_query_ranking_cases_but_not_profile_cases(golden, songs):
    ranking = evaluate_ranking(golden["ranking_cases"], songs, baseline_rank)
    agg = aggregate_runs([(ranking, [])])
    names = {c["name"] for c in agg["cases"]}
    assert "Query: fall asleep" in names
    assert "Starter lofi" not in names
    assert all(c["kind"] == "ranking query" for c in agg["cases"])


def test_multi_run_report_flags_unstable_cases_and_shows_ranges(songs):
    runs = run_parser_n_times(songs, flaky_parser(), 4)
    report = format_multi_run_report(runs, True, songs, label="gemini parser")
    assert "# Evaluation report (gemini parser, 4 runs)" in report
    assert "2/4 (unstable)" in report
    assert "Unstable between runs: relaxing acoustic folk" in report
    assert "0% to 100%" in report  # easy-parser accuracy swung between 0% and 100%


def test_multi_run_report_warns_about_a_small_number_of_runs(songs):
    report = format_multi_run_report(run_parser_n_times(songs, flaky_parser(), 3), True, songs)
    assert "Only 3 runs" in report


def test_multi_run_report_counts_fallbacks_across_all_runs(songs):
    def fell_back(query, genres, moods):
        return {"genre": "folk", "mood": "", "energy": 0.5}, [FALLBACK_NOTE_PREFIX + " (test)"]

    report = format_multi_run_report(run_parser_n_times(songs, fell_back, 3), True, songs)
    assert "| Answers that fell back to keywords | 3/3 |" in report
    assert "came from the keyword fallback" in report


def test_stable_deterministic_parser_shows_no_spread(songs):
    runs = run_parser_n_times(songs, lambda q, g, m: ({"genre": "folk", "mood": "", "energy": 0.5}, []), 5)
    report = format_multi_run_report(runs, True, songs)
    assert "(unstable)" not in report
    assert "Unstable between runs" not in report
    assert "Only 5 runs" not in report


def test_main_runs_option_writes_a_multi_run_report(tmp_path):
    out = tmp_path / "multi.md"
    assert main(["--runs", "3", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "(baseline, 3 runs)" in text
    assert "## Case stability" in text


@pytest.mark.parametrize("bad", ["0", "-1", str(MAX_RUNS + 1)])
def test_main_rejects_run_counts_outside_the_allowed_range(bad):
    with pytest.raises(SystemExit):
        main(["--runs", bad])
