"""Weight sensitivity and ablation study for the scoring rule.

Drops each feature in turn, tries equal weights, and nudges each weight by +/-50%.
Every variant is scored on the golden set's profile cases (deterministic, no API calls),
so the report shows which features carry the ranking and how fragile the chosen weights
are. Run `python -m src.ablation --out reports/ablation.md`.
"""

import argparse
import sys
from typing import Callable, Dict, List, Sequence, Tuple

from src.evaluation import DEFAULT_K, GOLDEN_PATH, SONGS_PATH, evaluate_ranking, load_golden, summarize
from src.guardrails import known_labels, sanitize_profile
from src.recommender import WEIGHTS, load_songs, recommend_songs

Weights = Dict[str, float]


def weighted_rank_fn(weights: Weights) -> Callable:
    """A rank_fn for evaluate_ranking that scores with the given weights."""

    def rank(profile: Dict, songs: List[Dict], k: int = DEFAULT_K) -> Tuple[List[int], List[str]]:
        genres, moods = known_labels(songs)
        clean, warnings = sanitize_profile(profile, genres, moods)
        return [song["id"] for song, _, _ in recommend_songs(clean, songs, k=k, weights=weights)], warnings

    return rank


def build_variants(base: Weights) -> List[Tuple[str, Weights]]:
    """The base weights plus every ablation and perturbation, in report order."""
    variants: List[Tuple[str, Weights]] = [("Baseline", dict(base))]
    for feature in base:
        variants.append((f"Without {feature}", {**base, feature: 0.0}))
    variants.append(("Equal weights", {feature: 1.0 for feature in base}))
    for feature in base:
        for factor in (0.5, 1.5):
            variants.append((f"{feature.capitalize()} x{factor}", {**base, feature: base[feature] * factor}))
    return variants


def evaluate_variants(
    cases: List[Dict], songs: List[Dict], base: Weights = None, k: int = DEFAULT_K
) -> List[Dict]:
    """Score each variant on the given profile cases."""
    base = dict(WEIGHTS if base is None else base)
    profile_cases = [c for c in cases if "profile" in c]
    results: List[Dict] = []
    baseline_top1: Dict[str, int] = {}

    for name, weights in build_variants(base):
        ranking = evaluate_ranking(profile_cases, songs, weighted_rank_fn(weights), k)
        summary = summarize(ranking, [])
        top1 = {r["name"]: (r["ranked"][0] if r["ranked"] else None) for r in ranking}
        if name == "Baseline":
            baseline_top1 = top1
        results.append({
            "name": name,
            "weights": weights,
            "passed": summary["ranking_cases_passed"],
            "total": summary["ranking_cases_total"],
            "top1_rate": summary["top1_rate"],
            "ndcg": summary["mean_ndcg"],
            "precision": summary["mean_precision"],
            "forbidden_hits": summary["forbidden_hits"],
            "top1_changed": sum(1 for case, song_id in top1.items() if song_id != baseline_top1.get(case)),
        })
    return results


def format_report(results: List[Dict], songs: List[Dict], k: int = DEFAULT_K) -> str:
    base = results[0]
    lines = [
        "# Weight sensitivity and ablation",
        "",
        f"Catalog: {len(songs)} songs. Scored on the {base['total']} profile cases in the golden set "
        "(free-text cases are excluded because they depend on the parser, not the weights).",
        "",
        "| Variant | Weights (energy / mood / genre / acoustic) | Cases passed | "
        f"Top-1 relevant | nDCG@{k} | P@{k} | Top-1 changed vs baseline | Forbidden songs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        w = r["weights"]
        weights = " / ".join(f"{w[f]:.2f}" for f in ("energy", "mood", "genre", "acoustic"))
        lines.append(
            f"| {r['name']} | {weights} | {r['passed']}/{r['total']} | {r['top1_rate']:.0%} | "
            f"{r['ndcg']:.2f} | {r['precision']:.2f} | {r['top1_changed']} | {r['forbidden_hits']} |"
        )

    ablations = [r for r in results if r["name"].startswith("Without ")]
    worst = min(ablations, key=lambda r: r["ndcg"])
    least = max(ablations, key=lambda r: r["ndcg"])
    perturbed = [r for r in results if " x" in r["name"]]
    lines += [
        "",
        "## Reading it",
        "",
        f"- Dropping a feature costs the most for **{worst['name'].removeprefix('Without ')}** "
        f"(nDCG {base['ndcg']:.2f} to {worst['ndcg']:.2f}) and the least for "
        f"**{least['name'].removeprefix('Without ')}** (to {least['ndcg']:.2f}).",
        f"- The weights are {'fairly robust' if all(abs(r['ndcg'] - base['ndcg']) <= 0.05 for r in perturbed) else 'sensitive'}: "
        f"the +/-50% variants range from nDCG {min(r['ndcg'] for r in perturbed):.2f} to "
        f"{max(r['ndcg'] for r in perturbed):.2f}.",
        "- These cases and their judgments were written by one person, so treat small differences as noise. "
        "Only large drops are evidence that a feature matters.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Weight sensitivity and ablation study")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", help="also write the report to this file")
    args = parser.parse_args(argv)

    songs = load_songs(SONGS_PATH)
    golden = load_golden(GOLDEN_PATH)
    report = format_report(evaluate_variants(golden["ranking_cases"], songs, k=args.k), songs, args.k)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
