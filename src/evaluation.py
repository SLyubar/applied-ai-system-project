"""Evaluation harness: score any ranking pipeline and query parser against the golden set.

Run `python -m src.evaluation` for the report. The pipelines are pluggable
(`rank_fn`, `parse_fn`) so the same golden set can compare the baseline with the
planned Claude parser and embedding retriever.
"""

import argparse
import json
import sys
from typing import Callable, Dict, List, Sequence, Tuple

from src.guardrails import known_labels, sanitize_profile
from src.parser import parse_query
from src.recommender import load_songs, recommend_songs

GOLDEN_PATH = "data/golden_set.json"
SONGS_PATH = "data/songs.csv"
DEFAULT_K = 5

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


def check_parse(profile: Dict, expected: Dict) -> List[str]:
    """Return the names of expected fields the parsed profile got wrong.

    expected may contain exact values (genre, mood, likes_acoustic) and bounds
    (energy_min, energy_max).
    """
    failures: List[str] = []
    for key in ("genre", "mood", "likes_acoustic"):
        if key in expected and profile.get(key) != expected[key]:
            failures.append(key)
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


def evaluate_ranking(cases: List[Dict], songs: List[Dict], rank_fn: RankFn, k: int = DEFAULT_K) -> List[Dict]:
    """Run every ranking case and return one result dict per case."""
    results: List[Dict] = []
    by_name: Dict[str, List[int]] = {}

    for case in cases:
        ranked, warnings = rank_fn(case["profile"], songs, k)
        by_name[case["name"]] = ranked
        relevant = case.get("relevant_ids", [])
        has_truth = bool(relevant)
        result = {
            "name": case["name"],
            "ranked": ranked,
            "warnings": warnings,
            "has_ground_truth": has_truth,
            "expect_warning": case.get("expect_warning", False),
            "equivalent_to": case.get("equivalent_to"),
        }
        if has_truth:
            result["precision"] = precision_at_k(ranked, relevant, k)
            result["rr"] = reciprocal_rank(ranked, relevant)
            result["top1_hit"] = bool(ranked) and ranked[0] in set(relevant)
        results.append(result)

    for result in results:
        checks: List[Tuple[str, bool]] = []
        if result["has_ground_truth"]:
            checks.append(("top-1 is relevant", result["top1_hit"]))
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
            profile, _ = parse_fn(case["query"], genres, moods)
            failed = check_parse(profile, case["expected"])
        except ValueError as error:
            profile, failed = {}, [f"error: {error}"]
        results.append({
            "query": case["query"],
            "tier": case.get("tier", "easy"),
            "profile": profile,
            "failed_fields": failed,
            "passed": not failed,
        })
    return results


def is_deterministic(cases: List[Dict], songs: List[Dict], rank_fn: RankFn, k: int = DEFAULT_K) -> bool:
    """True if running every case twice gives identical rankings and warnings."""
    return all(rank_fn(c["profile"], songs, k) == rank_fn(c["profile"], songs, k) for c in cases)


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
        "mrr": mean([r["rr"] for r in scored]),
        "top1_rate": mean([1.0 if r["top1_hit"] else 0.0 for r in scored]),
        "ranking_cases_passed": sum(1 for r in ranking if r["passed"]),
        "ranking_cases_total": len(ranking),
        "parser_overall": rate(parsing),
        "parser_easy": rate(easy),
        "parser_hard": rate(hard),
        "parser_passed": sum(1 for p in parsing if p["passed"]),
        "parser_total": len(parsing),
    }


# ---------------------------------------------------------------- report

def format_report(
    ranking: List[Dict],
    parsing: List[Dict],
    deterministic: bool,
    songs: List[Dict],
    k: int = DEFAULT_K,
    label: str = "baseline",
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
        f"| Mean reciprocal rank | {s['mrr']:.2f} |",
        f"| Top-1 relevant rate | {s['top1_rate']:.0%} |",
        f"| Ranking cases passed | {s['ranking_cases_passed']}/{s['ranking_cases_total']} |",
        f"| Parser accuracy (all) | {s['parser_passed']}/{s['parser_total']} ({s['parser_overall']:.0%}) |",
        f"| Parser accuracy (easy / hard) | {s['parser_easy']:.0%} / {s['parser_hard']:.0%} |",
        f"| Deterministic (repeat runs identical) | {'yes' if deterministic else 'NO'} |",
        "",
        "## Ranking cases",
        "",
        f"| Case | P@{k} | RR | Result | Top 3 |",
        "|---|---|---|---|---|",
    ]
    for r in ranking:
        precision = f"{r['precision']:.2f}" if r["has_ground_truth"] else "n/a"
        rr = f"{r['rr']:.2f}" if r["has_ground_truth"] else "n/a"
        top3 = "; ".join(titles.get(i, str(i)) for i in r["ranked"][:3])
        lines.append(f"| {r['name']} | {precision} | {rr} | {'PASS' if r['passed'] else 'FAIL'} | {top3} |")

    failed_ranking = [r for r in ranking if not r["passed"]]
    if failed_ranking:
        lines += ["", "### Failed ranking checks", ""]
        for r in failed_ranking:
            bad = ", ".join(name for name, ok in r["checks"] if not ok)
            lines.append(f"- **{r['name']}**: {bad}")

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
    if no_truth:
        lines.append(f"- Cases with no ground truth, checked only for warnings: {', '.join(no_truth)}.")
    if failed_ranking or any(not p["passed"] for p in parsing):
        lines.append("- Decide for each failure whether the system or the golden set is wrong.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] = None) -> int:
    arg_parser = argparse.ArgumentParser(description="Evaluate the recommender against the golden set")
    arg_parser.add_argument("--k", type=int, default=DEFAULT_K, help="cutoff for precision@k")
    arg_parser.add_argument("--out", help="also write the report to this file")
    arg_parser.add_argument("--label", default="baseline", help="name shown in the report title")
    arg_parser.add_argument("--strict", action="store_true", help="exit 1 if any case fails")
    args = arg_parser.parse_args(argv)

    songs = load_songs(SONGS_PATH)
    golden = load_golden()

    ranking = evaluate_ranking(golden["ranking_cases"], songs, baseline_rank, args.k)
    parsing = evaluate_parser(golden["parser_cases"], songs, parse_query)
    deterministic = is_deterministic(golden["ranking_cases"], songs, baseline_rank, args.k)

    report = format_report(ranking, parsing, deterministic, songs, args.k, args.label)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)

    all_passed = all(r["passed"] for r in ranking) and all(p["passed"] for p in parsing) and deterministic
    return 0 if (all_passed or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
