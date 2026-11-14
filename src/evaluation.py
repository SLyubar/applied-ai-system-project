"""Evaluation harness: score any ranking pipeline and query parser against the golden set.

Run `python -m src.evaluation` for the report. The pipelines are pluggable
(`rank_fn`, `parse_fn`) so the same golden set can compare the baseline with the
Gemini parser and the planned embedding retriever.
"""

import argparse
import json
import math
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

from src.confidence import assess, calibration_table, understanding
from src.guardrails import known_labels, sanitize_profile
from src.parser import get_parse_fn
from src.recommender import load_songs, recommend_songs, score_song

FALLBACK_NOTE_PREFIX = "Gemini parser unavailable"  # see src/gemini_parser.py
GOLDEN_PATH = "data/golden_set.json"
SONGS_PATH = "data/songs.csv"
DEFAULT_K = 5
MAX_RUNS = 20  # each live run makes dozens of API calls (see estimate_api_calls); cap accidental spending

# rank_fn(raw_profile, songs, k) -> (ranked song ids, warnings)
RankFn = Callable[[Dict, List[Dict], int], Tuple[List[int], List[str]]]
# parse_fn(query, known_genres, known_moods) -> (profile, notes)
ParseFn = Callable[[str, Sequence[str], Sequence[str]], Tuple[Dict, List[str]]]


# ---------------------------------------------------------------- metrics

def precision_at_k(ranked: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Fraction of the top k results that are relevant (denominator is always k)."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    return sum(1 for song_id in ranked[:k] if song_id in relevant_set) / k


def reciprocal_rank(ranked: Sequence[int], relevant: Sequence[int]) -> float:
    """1 / rank of the first relevant result, or 0.0 if none appears."""
    relevant_set = set(relevant)
    for position, song_id in enumerate(ranked, start=1):
        if song_id in relevant_set:
            return 1.0 / position
    return 0.0


IDEAL_GRADE = 2       # a song the listener would be delighted to get
ACCEPTABLE_GRADE = 1  # a song the listener would be fine with


def case_grades(case: Dict) -> Dict[int, int]:
    """Map song id -> relevance grade for a ranking case (ideal beats acceptable)."""
    grades = {song_id: ACCEPTABLE_GRADE for song_id in case.get("acceptable_ids", [])}
    grades.update({song_id: IDEAL_GRADE for song_id in case.get("ideal_ids", [])})
    return grades


def ndcg_at_k(ranked: Sequence[int], grades: Dict[int, int], k: int) -> float:
    """Normalized discounted cumulative gain: 1.0 means the best possible top k.

    Uses the grade itself as the gain and a 1/log2(rank + 1) position discount, so
    putting an ideal song first counts for more than putting it fifth.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not grades:
        return 0.0

    def dcg(gains: Sequence[int]) -> float:
        return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))

    actual = dcg([grades.get(song_id, 0) for song_id in ranked[:k]])
    best = dcg(sorted(grades.values(), reverse=True)[:k])
    return actual / best if best else 0.0


def forbidden_in_top_k(ranked: Sequence[int], forbidden: Sequence[int], k: int) -> List[int]:
    """Songs that must not be recommended but appear in the top k."""
    banned = set(forbidden)
    return [song_id for song_id in ranked[:k] if song_id in banned]


def check_parse(profile: Dict, expected: Dict) -> List[str]:
    """Return the names of expected fields the parsed profile got wrong.

    expected may contain exact values (genre, mood, likes_acoustic), allowed sets
    (genre_in, mood_in), and bounds (energy_min, energy_max).
    """
    failures: List[str] = []
    for key in ("genre", "mood", "likes_acoustic"):
        if key in expected and profile.get(key) != expected[key]:
            failures.append(key)
    for key in ("genre", "mood"):
        if f"{key}_in" in expected and profile.get(key) not in expected[f"{key}_in"]:
            failures.append(f"{key}_in")
    if "energy_min" in expected and not profile.get("energy", 0.0) >= expected["energy_min"]:
        failures.append("energy_min")
    if "energy_max" in expected and not profile.get("energy", 1.0) <= expected["energy_max"]:
        failures.append("energy_max")
    return failures


# ---------------------------------------------------------------- pipelines

def baseline_rank(profile: Dict, songs: List[Dict], k: int = DEFAULT_K) -> Tuple[List[int], List[str]]:
    """The current pipeline: guardrails, then the deterministic scorer."""
    genres, moods = known_labels(songs)
    clean, warnings = sanitize_profile(profile, genres, moods)
    ids = [song["id"] for song, _, _ in recommend_songs(clean, songs, k=k)]
    return ids, warnings


# ---------------------------------------------------------------- evaluation

def load_golden(path: str = GOLDEN_PATH) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ranking(
    cases: List[Dict],
    songs: List[Dict],
    rank_fn: RankFn,
    k: int = DEFAULT_K,
    parse_fn: ParseFn = None,
) -> List[Dict]:
    """Run every ranking case and return one result dict per case.

    A case supplies either a ready `profile` or a free-text `query`. Queries are
    parsed with `parse_fn` (the keyword parser by default), so those cases test the
    whole pipeline end to end.

    A case passes only if every applicable check holds. These criteria are fixed in
    advance, not tuned to results:
      - top-1 is a relevant song (ideal or acceptable)
      - no forbidden song appears in the top k
      - at least one ideal song appears in the top 3 (when the case lists ideal songs)
      - the input produced a warning (when expect_warning is set)
      - the ranking equals another case's ranking (when equivalent_to is set)
    """
    if parse_fn is None:
        parse_fn = get_parse_fn("keyword")
    genres, moods = known_labels(songs)
    results: List[Dict] = []
    by_name: Dict[str, List[int]] = {}

    songs_by_id = {song["id"]: song for song in songs}
    for case in cases:
        fell_back = False
        parse_error = None
        notes: List[str] = []
        if "query" in case:
            try:
                profile, notes = parse_fn(case["query"], genres, moods)
                fell_back = any(n.startswith(FALLBACK_NOTE_PREFIX) for n in notes)
            except ValueError as error:
                profile, parse_error = None, str(error)
        else:
            profile = case["profile"]

        if profile is None:
            ranked, warnings = [], [f"query could not be parsed: {parse_error}"]
        elif "query" in case and getattr(rank_fn, "wants_query", False):
            # Retrieval-aware pipelines also need the original text, not just the parsed profile.
            ranked, warnings = rank_fn(profile, songs, k, query=case["query"])
        else:
            ranked, warnings = rank_fn(profile, songs, k)
        by_name[case["name"]] = ranked

        confidence = None
        if profile is not None and ranked and ranked[0] in songs_by_id:
            try:
                clean, _ = sanitize_profile(profile, genres, moods)
                top_score = score_song(clean, songs_by_id[ranked[0]])[0]
                confidence = assess(profile, notes, genres, moods, top_score).as_dict()
            except ValueError:
                confidence = None  # a profile too broken to clean has no meaningful confidence

        grades = case_grades(case)
        has_truth = bool(grades)
        result = {
            "name": case["name"],
            "ranked": ranked,
            "warnings": warnings,
            "has_ground_truth": has_truth,
            "is_query": "query" in case,
            "fell_back": fell_back,
            "expect_warning": case.get("expect_warning", False),
            "equivalent_to": case.get("equivalent_to"),
            "forbidden_hits": forbidden_in_top_k(ranked, case.get("forbidden_ids", []), k),
            "confidence": confidence,
        }
        if has_truth:
            result["precision"] = precision_at_k(ranked, list(grades), k)
            result["rr"] = reciprocal_rank(ranked, list(grades))
            result["ndcg"] = ndcg_at_k(ranked, grades, k)
            result["top1_hit"] = bool(ranked) and ranked[0] in grades
            ideal = set(case.get("ideal_ids", []))
            result["ideal_in_top3"] = bool(ideal & set(ranked[:3])) if ideal else None
        results.append(result)

    for case, result in zip(cases, results):
        checks: List[Tuple[str, bool]] = []
        if result["has_ground_truth"]:
            checks.append(("top-1 is relevant", result["top1_hit"]))
        if case.get("forbidden_ids"):
            checks.append((
                "no forbidden songs in top k" if not result["forbidden_hits"]
                else f"forbidden songs in top k: {result['forbidden_hits']}",
                not result["forbidden_hits"],
            ))
        if result.get("ideal_in_top3") is not None:
            checks.append(("an ideal song is in the top 3", result["ideal_in_top3"]))
        if result["expect_warning"]:
            checks.append(("warns about the input", bool(result["warnings"])))
        if result["equivalent_to"]:
            same = result["ranked"] == by_name.get(result["equivalent_to"])
            checks.append((f"ranks like {result['equivalent_to']!r}", same))
        result["checks"] = checks
        result["passed"] = all(ok for _, ok in checks)

    return results


def evaluate_parser(cases: List[Dict], songs: List[Dict], parse_fn: ParseFn) -> List[Dict]:
    genres, moods = known_labels(songs)
    results: List[Dict] = []
    for case in cases:
        try:
            profile, notes = parse_fn(case["query"], genres, moods)
            failed = check_parse(profile, case["expected"])
        except ValueError as error:
            profile, notes, failed = {}, [], [f"error: {error}"]
        results.append({
            "query": case["query"],
            "tier": case.get("tier", "easy"),
            "profile": profile,
            "failed_fields": failed,
            "passed": not failed,
            "understanding": understanding(profile, notes, genres, moods)[0] if profile else 0.0,
            # A fallback answer must not be credited to the parser under test.
            "fell_back": any(n.startswith(FALLBACK_NOTE_PREFIX) for n in notes),
        })
    return results


def calibration_lines(ranking_runs: List[List[Dict]], parsing_runs: List[List[Dict]]) -> List[str]:
    """Markdown checking that the confidence score means something.

    Groups the ranking cases that have ground truth by the confidence band the system
    reported, and shows each band's pass rate. A useful score has pass rates that fall from
    high to low. Also compares the parser's understanding score on phrases it got right and wrong.
    """
    rows = [
        (r["confidence"]["label"], r["passed"])
        for run in ranking_runs for r in run
        if r["has_ground_truth"] and r.get("confidence")
    ]
    runs = len(ranking_runs)
    lines = [
        "",
        "## Confidence calibration",
        "",
        "Does the confidence score separate good results from bad? Ranking cases with ground truth, grouped by "
        "the confidence band the system reported" + (f", pooled over {runs} runs." if runs > 1 else "."),
        "",
        "| Confidence | Cases | Passed | Pass rate |",
        "|---|---|---|---|",
    ]
    table = calibration_table(rows)
    for t in table:
        rate = f"{t['pass_rate']:.0%}" if t["pass_rate"] is not None else "n/a"
        lines.append(f"| {t['band']} | {t['cases']} | {t['passed']} | {rate} |")
    rates = [t["pass_rate"] for t in table if t["pass_rate"] is not None]
    if len(rates) < 2:
        verdict = "Too few confidence bands are populated to judge calibration."
    elif rates[0] > rates[-1] and all(a >= b for a, b in zip(rates, rates[1:])):
        verdict = "Pass rates fall as confidence falls, as they should."
    else:
        verdict = "Pass rates do not fall cleanly with confidence, so treat the score as a rough guide only."
    lines += ["", verdict]

    def mean(values: List[float]) -> float:
        return sum(values) / len(values)

    right = [p["understanding"] for run in parsing_runs for p in run if p["passed"] and "understanding" in p]
    wrong = [p["understanding"] for run in parsing_runs for p in run if not p["passed"] and "understanding" in p]
    if right and wrong:
        lines.append(
            f"Parser understanding score: mean {mean(right):.2f} on phrases it got right ({len(right)}) versus "
            f"{mean(wrong):.2f} on phrases it got wrong ({len(wrong)})."
        )
    return lines


def estimate_api_calls(golden: Dict) -> int:
    """Parser calls one evaluation run makes: every free-text case and phrase is parsed once per run."""
    return (
        len(golden["parser_cases"])
        + sum(1 for c in golden["ranking_cases"] if "query" in c)
        + sum(len(g["queries"]) for g in golden.get("paraphrase_groups", []))
        + len(golden.get("adversarial_cases", []))
    )


def is_deterministic(cases: List[Dict], songs: List[Dict], rank_fn: RankFn, k: int = DEFAULT_K) -> bool:
    """True if running every profile case twice gives identical rankings and warnings.

    Query cases are skipped: with a live LLM parser they are not expected to repeat exactly.
    """
    profile_cases = [c for c in cases if "profile" in c]
    return all(rank_fn(c["profile"], songs, k) == rank_fn(c["profile"], songs, k) for c in profile_cases)


def summarize(ranking: List[Dict], parsing: List[Dict]) -> Dict:
    scored = [r for r in ranking if r["has_ground_truth"]]
    easy = [p for p in parsing if p["tier"] == "easy"]
    hard = [p for p in parsing if p["tier"] == "hard"]

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def rate(items: List[Dict]) -> float:
        return mean([1.0 if i["passed"] else 0.0 for i in items])

    return {
        "mean_precision": mean([r["precision"] for r in scored]),
        "mean_ndcg": mean([r["ndcg"] for r in scored]),
        "mrr": mean([r["rr"] for r in scored]),
        "top1_rate": mean([1.0 if r["top1_hit"] else 0.0 for r in scored]),
        "ranking_cases_passed": sum(1 for r in ranking if r["passed"]),
        "ranking_cases_total": len(ranking),
        "parser_overall": rate(parsing),
        "parser_easy": rate(easy),
        "parser_hard": rate(hard),
        "parser_passed": sum(1 for p in parsing if p["passed"]),
        "parser_total": len(parsing),
        "parser_fallbacks": sum(1 for p in parsing if p.get("fell_back")),
        "ranking_query_cases": sum(1 for r in ranking if r.get("is_query")),
        "ranking_fallbacks": sum(1 for r in ranking if r.get("fell_back")),
        "forbidden_hits": sum(len(r["forbidden_hits"]) for r in ranking),
    }


def _latency_rows(seconds_per_run: Sequence[float]) -> List[str]:
    """Wall-clock time per run (parser calls included). Cost is not tracked."""
    if not seconds_per_run:
        return []
    mean = sum(seconds_per_run) / len(seconds_per_run)
    return [f"| Wall time per run | {mean:.1f}s (min {min(seconds_per_run):.1f}s, max {max(seconds_per_run):.1f}s) |"]


# ---------------------------------------------------------------- repeated runs

# (row label, key in summarize(), how to format)
MULTI_RUN_ROWS = [
    ("Ranking cases passed", "ranking_cases_passed", "count"),
    ("Top-1 relevant rate", "top1_rate", "pct"),
    ("Mean nDCG", "mean_ndcg", "dec"),
    ("Mean precision", "mean_precision", "dec"),
    ("Mean reciprocal rank", "mrr", "dec"),
    ("Parser phrases passed", "parser_passed", "count"),
    ("Parser accuracy, easy", "parser_easy", "pct"),
    ("Parser accuracy, hard", "parser_hard", "pct"),
    ("Forbidden songs recommended", "forbidden_hits", "count"),
]


def _format_stat(values: List[float], kind: str) -> str:
    mean, low, high = sum(values) / len(values), min(values), max(values)
    if kind == "pct":
        return f"{mean:.0%} ({low:.0%} to {high:.0%})"
    if kind == "count":
        return f"{mean:.1f} ({low:g} to {high:g})"
    return f"{mean:.2f} ({low:.2f} to {high:.2f})"


def aggregate_runs(runs: List[Tuple[List[Dict], List[Dict]]]) -> Dict:
    """Combine several (ranking, parsing) runs into per-metric lists and per-case pass counts.

    Returns {"summaries": [summarize() per run], "cases": [...]}, where each case entry
    has its kind ("parser" or "ranking query"), name, passes, runs, and fallbacks. Only
    cases that depend on the parser appear; profile cases never use it.
    """
    summaries = [summarize(ranking, parsing) for ranking, parsing in runs]
    cases: Dict[Tuple[str, str], Dict] = {}

    def record(kind: str, name: str, passed: bool, fell_back: bool) -> None:
        entry = cases.setdefault((kind, name), {"kind": kind, "name": name, "passes": 0, "runs": 0, "fallbacks": 0})
        entry["runs"] += 1
        entry["passes"] += 1 if passed else 0
        entry["fallbacks"] += 1 if fell_back else 0

    for ranking, parsing in runs:
        for p in parsing:
            record("parser", p["query"], p["passed"], p.get("fell_back", False))
        for r in ranking:
            if r.get("is_query"):
                record("ranking query", r["name"], r["passed"], r.get("fell_back", False))
    return {"summaries": summaries, "cases": list(cases.values())}


def format_multi_run_report(
    runs: List[Tuple[List[Dict], List[Dict]]],
    deterministic: bool,
    songs: List[Dict],
    k: int = DEFAULT_K,
    label: str = "baseline",
    seconds_per_run: Sequence[float] = (),
) -> str:
    n = len(runs)
    agg = aggregate_runs(runs)
    summaries = agg["summaries"]
    first_ranking, first_parsing = runs[0]
    fallbacks = sum(s["parser_fallbacks"] + s["ranking_fallbacks"] for s in summaries)
    possible = n * (summaries[0]["parser_total"] + summaries[0]["ranking_query_cases"])

    lines: List[str] = [
        f"# Evaluation report ({label}, {n} runs)",
        "",
        f"Catalog: {len(songs)} songs. Golden set: {len(first_ranking)} ranking cases, {len(first_parsing)} parser cases. "
        f"Each cell shows the mean over {n} runs, with the range in parentheses.",
        "",
        "## Summary",
        "",
        "| Metric | Mean (min to max) |",
        "|---|---|",
    ]
    for row_label, key, kind in MULTI_RUN_ROWS:
        suffix = f"@{k}" if key in ("mean_ndcg", "mean_precision") else ""
        suffix += {"ranking_cases_passed": f" (of {summaries[0]['ranking_cases_total']})",
                   "parser_passed": f" (of {summaries[0]['parser_total']})"}.get(key, "")
        lines.append(f"| {row_label}{suffix} | {_format_stat([s[key] for s in summaries], kind)} |")
    lines += [
        f"| Deterministic profile cases (repeat runs identical) | {'yes' if deterministic else 'NO'} |",
        f"| Answers that fell back to keywords | {fallbacks}/{possible} |",
    ] + _latency_rows(seconds_per_run) + [
        "",
        "## Case stability",
        "",
        "Only cases that use the parser can vary between runs. Unstable cases are listed first.",
        "",
        "| Case | Type | Passed | Fell back |",
        "|---|---|---|---|",
    ]
    ordered = sorted(agg["cases"], key=lambda c: (c["passes"] in (0, c["runs"]), c["passes"] / c["runs"], c["name"]))
    for c in ordered:
        mark = "" if c["passes"] in (0, c["runs"]) else " (unstable)"
        lines.append(f"| {c['name']} | {c['kind']} | {c['passes']}/{c['runs']}{mark} | {c['fallbacks']}/{c['runs']} |")

    unstable = [c["name"] for c in agg["cases"] if 0 < c["passes"] < c["runs"]]
    always_fail = [c["name"] for c in agg["cases"] if c["passes"] == 0]
    lines += calibration_lines([r for r, _ in runs], [p for _, p in runs])
    lines += ["", "## Needs human review", "",
              "- The golden set is a set of human judgments. Check `data/golden_set.json` before trusting any number above."]
    if fallbacks:
        lines.append(
            f"- **{fallbacks} answer(s) came from the keyword fallback, not Gemini.** "
            "The numbers above do not measure Gemini alone. Check credentials, quota and network, then re-run."
        )
    if unstable:
        lines.append(f"- Unstable between runs: {'; '.join(unstable)}.")
    if always_fail:
        lines.append(f"- Failed in every run: {'; '.join(always_fail)}.")
    if n < 5:
        lines.append(f"- Only {n} runs: ranges are rough. Use 5 or more for a meaningful spread.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- report

def format_report(
    ranking: List[Dict],
    parsing: List[Dict],
    deterministic: bool,
    songs: List[Dict],
    k: int = DEFAULT_K,
    label: str = "baseline",
    seconds_per_run: Sequence[float] = (),
) -> str:
    titles = {s["id"]: s["title"] for s in songs}
    s = summarize(ranking, parsing)
    lines: List[str] = [
        f"# Evaluation report ({label})",
        "",
        f"Catalog: {len(songs)} songs. Golden set: {len(ranking)} ranking cases, {len(parsing)} parser cases.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean precision@{k} | {s['mean_precision']:.2f} |",
        f"| Mean nDCG@{k} | {s['mean_ndcg']:.2f} |",
        f"| Mean reciprocal rank | {s['mrr']:.2f} |",
        f"| Top-1 relevant rate | {s['top1_rate']:.0%} |",
        f"| Ranking cases passed | {s['ranking_cases_passed']}/{s['ranking_cases_total']} |",
        f"| Parser accuracy (all) | {s['parser_passed']}/{s['parser_total']} ({s['parser_overall']:.0%}) |",
        f"| Parser accuracy (easy / hard) | {s['parser_easy']:.0%} / {s['parser_hard']:.0%} |",
        f"| Deterministic (repeat runs identical) | {'yes' if deterministic else 'NO'} |",
        f"| Forbidden songs recommended (top {k}) | {s['forbidden_hits']} |",
        f"| Answers that fell back to keywords | "
        f"{s['parser_fallbacks'] + s['ranking_fallbacks']}/{s['parser_total'] + s['ranking_query_cases']} |",
    ] + _latency_rows(seconds_per_run) + [
        "",
        "## Ranking cases",
        "",
        f"| Case | Input | P@{k} | nDCG@{k} | Result | Top 3 |",
        "|---|---|---|---|---|---|",
    ]
    for r in ranking:
        precision = f"{r['precision']:.2f}" if r["has_ground_truth"] else "n/a"
        ndcg = f"{r['ndcg']:.2f}" if r["has_ground_truth"] else "n/a"
        top3 = "; ".join(titles.get(i, str(i)) for i in r["ranked"][:3])
        kind = "query" if r["is_query"] else "profile"
        lines.append(
            f"| {r['name']} | {kind} | {precision} | {ndcg} | {'PASS' if r['passed'] else 'FAIL'} | {top3} |"
        )

    failed_ranking = [r for r in ranking if not r["passed"]]
    if failed_ranking:
        lines += ["", "### Failed ranking checks", ""]
        for r in failed_ranking:
            bad = ", ".join(name for name, ok in r["checks"] if not ok)
            lines.append(f"- **{r['name']}**: {bad}")

    lines += calibration_lines([ranking], [parsing])

    lines += ["", "## Parser cases", "", "| Query | Tier | Result | Wrong fields |", "|---|---|---|---|"]
    for p in parsing:
        wrong = ", ".join(p["failed_fields"]) or "-"
        lines.append(f"| {p['query']} | {p['tier']} | {'PASS' if p['passed'] else 'FAIL'} | {wrong} |")

    no_truth = [r["name"] for r in ranking if not r["has_ground_truth"]]
    lines += [
        "",
        "## Needs human review",
        "",
        "- The golden set is a set of human judgments. Check `data/golden_set.json` before trusting any number above.",
    ]
    fallbacks = s["parser_fallbacks"] + s["ranking_fallbacks"]
    if fallbacks:
        lines.append(
            f"- **{fallbacks} answer(s) came from the keyword fallback, not Gemini.** "
            "The numbers above do not measure Gemini alone. Check credentials and network, then re-run."
        )
    if no_truth:
        lines.append(f"- Cases with no ground truth, checked only for warnings: {', '.join(no_truth)}.")
    if failed_ranking or any(not p["passed"] for p in parsing):
        lines.append("- Decide for each failure whether the system or the golden set is wrong.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] = None) -> int:
    arg_parser = argparse.ArgumentParser(description="Evaluate the recommender against the golden set")
    arg_parser.add_argument("--k", type=int, default=DEFAULT_K, help="cutoff for precision@k")
    arg_parser.add_argument("--out", help="also write the report to this file")
    arg_parser.add_argument("--label", help="name shown in the report title")
    arg_parser.add_argument("--strict", action="store_true", help="exit 1 if any case fails")
    arg_parser.add_argument(
        "--retriever", choices=["none", "tfidf", "gemini"], default="none",
        help="retrieve candidate songs by description before scoring (affects free-text cases only)",
    )
    arg_parser.add_argument("--pool", type=int, default=20, help="candidates the retriever passes to the scorer")
    arg_parser.add_argument(
        "--runs", type=int, default=1,
        help=f"repeat the whole evaluation N times (1 to {MAX_RUNS}) and report mean and range; "
             "useful for a non-deterministic parser. Live runs announce how many API calls they will make.",
    )
    arg_parser.add_argument(
        "--parser", choices=["keyword", "gemini"], default="keyword",
        help="query parser to evaluate; 'gemini' makes live API calls (needs credentials)",
    )
    args = arg_parser.parse_args(argv)
    if not 1 <= args.runs <= MAX_RUNS:
        arg_parser.error(f"--runs must be between 1 and {MAX_RUNS}")
    if args.label is None:
        args.label = "baseline" if args.parser == "keyword" else f"{args.parser} parser"
        if args.retriever != "none":
            args.label += f" + {args.retriever} retrieval (pool {args.pool})"

    songs = load_songs(SONGS_PATH)
    golden = load_golden()

    from src import robustness  # imported here: robustness itself imports from this module

    if args.parser != "keyword":
        per_run = estimate_api_calls(golden)
        print(
            f"This will make about {per_run * args.runs} {args.parser} API calls ({per_run} per run). "
            "Free tiers have a daily request limit per model (500 when this was written), and the "
            "limit is shared with every other run that day.",
            file=sys.stderr, flush=True,
        )
    parse_fn = get_parse_fn(args.parser)
    rank_fn = baseline_rank
    if args.retriever != "none":
        from src.retrieval import Retriever, get_embedder, make_retrieval_rank_fn

        try:
            rank_fn = make_retrieval_rank_fn(Retriever(songs, get_embedder(args.retriever)), args.pool)
        except Exception as error:  # evaluating a retriever that cannot run would be meaningless
            arg_parser.error(f"cannot build the {args.retriever} retriever: {type(error).__name__}: {error}")
    runs = []
    robustness_runs = []
    seconds_per_run = []
    for _ in range(args.runs):
        started = time.perf_counter()
        runs.append((
            evaluate_ranking(golden["ranking_cases"], songs, rank_fn, args.k, parse_fn),
            evaluate_parser(golden["parser_cases"], songs, parse_fn),
        ))
        robustness_runs.append({
            "paraphrase": robustness.evaluate_paraphrases(golden.get("paraphrase_groups", []), songs, parse_fn),
            "adversarial": robustness.evaluate_adversarial(golden.get("adversarial_cases", []), songs, parse_fn),
        })
        seconds_per_run.append(time.perf_counter() - started)
        if args.parser != "keyword" or args.runs > 1:
            # Progress on stderr so a slow or stalled live run is visible, not silent.
            print(f"run {len(runs)}/{args.runs} finished in {seconds_per_run[-1]:.0f}s", file=sys.stderr, flush=True)
    deterministic = is_deterministic(golden["ranking_cases"], songs, rank_fn, args.k)

    if args.runs == 1:
        report = format_report(runs[0][0], runs[0][1], deterministic, songs, args.k, args.label, seconds_per_run)
    else:
        report = format_multi_run_report(runs, deterministic, songs, args.k, args.label, seconds_per_run)
    report += "\n" + robustness.format_section(robustness_runs)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)

    # --strict requires every run to pass everything.
    all_passed = deterministic and all(
        all(r["passed"] for r in ranking) and all(p["passed"] for p in parsing) for ranking, parsing in runs
    ) and all(
        all(r["passed"] for r in run["paraphrase"]) and all(r["passed"] for r in run["adversarial"])
        for run in robustness_runs
    )
    return 0 if (all_passed or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
