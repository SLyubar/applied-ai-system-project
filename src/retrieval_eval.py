"""Does retrieval help? Compares the baseline against retrieval configurations.

Run on the golden set's free-text ranking cases. For each configuration it reports the
end-to-end ranking quality, plus how well the retriever alone does its job:

- relevant recall: the share of ideal + acceptable songs that land in the candidate pool
  (a song the retriever drops can never be recommended);
- ideal recall: the same for ideal songs only;
- forbidden candidates: how many must-not-recommend songs the pool lets through.

    python -m src.retrieval_eval --out reports/retrieval.md
    python -m src.retrieval_eval --parser gemini --embedders tfidf,gemini

The gemini embedder and parser make live API calls (embeddings are cached on disk).
"""

import argparse
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.evaluation import (
    DEFAULT_K,
    baseline_rank,
    case_grades,
    evaluate_ranking,
    load_golden,
    summarize,
    SONGS_PATH,
)
from src.parser import get_parse_fn
from src.recommender import load_songs
from src.retrieval import Retriever, get_embedder, make_retrieval_rank_fn

POOL_SIZES = (10, 20, 40)


def retrieval_only_rank_fn(retriever: Retriever) -> Callable:
    """Rank purely by description similarity, with no scorer. Shows what retrieval alone can do."""

    def rank(profile: Dict, songs: List[Dict], k: int = DEFAULT_K, query: Optional[str] = None):
        if not query:
            return baseline_rank(profile, songs, k)
        return [song_id for song_id, _ in retriever.search(query, k)], []

    rank.wants_query = True
    return rank


def candidate_metrics(retriever: Retriever, query_cases: List[Dict], pool: int) -> Dict:
    """How well the candidate pool of the given size covers each case's relevant songs."""
    relevant_recall, ideal_recall, forbidden = [], [], 0
    for case in query_cases:
        grades = case_grades(case)
        if not grades:
            continue
        candidates = {song_id for song_id, _ in retriever.search(case["query"], pool)}
        relevant_recall.append(len(candidates & set(grades)) / len(grades))
        ideal = set(case.get("ideal_ids", []))
        if ideal:
            ideal_recall.append(len(candidates & ideal) / len(ideal))
        forbidden += len(candidates & set(case.get("forbidden_ids", [])))

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {"relevant_recall": mean(relevant_recall), "ideal_recall": mean(ideal_recall), "forbidden_candidates": forbidden}


def run_config(name: str, rank_fn: Callable, query_cases: List[Dict], songs: List[Dict], parse_fn, k: int, cand: Optional[Dict]) -> Dict:
    ranking = evaluate_ranking(query_cases, songs, rank_fn, k, parse_fn)
    s = summarize(ranking, [])
    return {
        "name": name,
        "passed": s["ranking_cases_passed"],
        "total": s["ranking_cases_total"],
        "top1": s["top1_rate"],
        "ndcg": s["mean_ndcg"],
        "precision": s["mean_precision"],
        "forbidden": s["forbidden_hits"],
        "fallbacks": s["ranking_fallbacks"],
        "candidates": cand,
    }


def memoize_parser(parse_fn) -> Callable:
    """Parse each query once and reuse the answer, so every configuration sees identical profiles.

    This isolates the effect of retrieval from run-to-run variation in an LLM parser, and
    keeps a live experiment to one parser call per query.
    """
    cache: Dict[str, Tuple[Dict, List[str]]] = {}

    def parse(query: str, genres, moods):
        if query not in cache:
            cache[query] = parse_fn(query, genres, moods)
        profile, notes = cache[query]
        return dict(profile), list(notes)

    return parse


def run_experiment(
    songs: List[Dict],
    cases: List[Dict],
    parse_fn,
    embedder_names: Sequence[str] = ("tfidf",),
    pool_sizes: Sequence[int] = POOL_SIZES,
    k: int = DEFAULT_K,
    embedder_factory: Callable = get_embedder,
) -> Tuple[List[Dict], List[str]]:
    """Run every configuration. Returns (rows, notes); notes explain anything skipped."""
    query_cases = [c for c in cases if "query" in c]
    parse_fn = memoize_parser(parse_fn)  # one parse per query, shared by every configuration
    rows = [run_config("No retrieval (baseline)", baseline_rank, query_cases, songs, parse_fn, k, None)]
    notes: List[str] = []

    for embedder_name in embedder_names:
        embedder_rows: List[Dict] = []
        try:
            retriever = Retriever(songs, embedder_factory(embedder_name))
            embedder_rows.append(run_config(
                f"{embedder_name}: retrieval only (no scorer)", retrieval_only_rank_fn(retriever),
                query_cases, songs, parse_fn, k, candidate_metrics(retriever, query_cases, k),
            ))
            for pool in pool_sizes:
                embedder_rows.append(run_config(
                    f"{embedder_name}: retrieve {pool}, then score", make_retrieval_rank_fn(retriever, pool),
                    query_cases, songs, parse_fn, k, candidate_metrics(retriever, query_cases, pool),
                ))
        except Exception as error:  # a missing key, quota or network failure must not abort the whole report
            notes.append(f"{embedder_name} embedder skipped: {type(error).__name__}: {' '.join(str(error).split())[:140]}")
            continue
        rows.extend(embedder_rows)
    return rows, notes


def format_report(rows: List[Dict], notes: List[str], songs: List[Dict], parser_name: str, k: int = DEFAULT_K) -> str:
    total = rows[0]["total"]
    lines = [
        f"# Retrieval experiment (parser: {parser_name})",
        "",
        f"Catalog: {len(songs)} songs. {total} free-text ranking cases. Every configuration reuses the same parsed "
        "profile for each query, so differences come from retrieval alone.",
        "",
        f"| Configuration | Cases passed | Top-1 relevant | nDCG@{k} | P@{k} | Forbidden in top {k} | "
        "Relevant recall of pool | Ideal recall of pool | Forbidden in pool |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["candidates"]
        pool_cols = (
            f"{c['relevant_recall']:.0%} | {c['ideal_recall']:.0%} | {c['forbidden_candidates']}" if c else "n/a | n/a | n/a"
        )
        lines.append(
            f"| {r['name']} | {r['passed']}/{r['total']} | {r['top1']:.0%} | {r['ndcg']:.2f} | "
            f"{r['precision']:.2f} | {r['forbidden']} | {pool_cols} |"
        )

    base = rows[0]
    scored = [r for r in rows[1:] if "then score" in r["name"]]
    lines += ["", "## Reading it", ""]
    if scored:
        best = max(scored, key=lambda r: r["ndcg"])
        delta = best["ndcg"] - base["ndcg"]
        verdict = (
            f"helps by {delta:+.2f} nDCG" if delta > 0.02 else
            f"hurts by {delta:+.2f} nDCG" if delta < -0.02 else
            f"makes no meaningful difference ({delta:+.2f} nDCG)"
        )
        lines.append(f"- The best retrieve-then-score configuration is **{best['name']}**, which {verdict} versus the baseline.")
    only = [r for r in rows if "retrieval only" in r["name"]]
    for r in only:
        lines.append(f"- {r['name']}: nDCG {r['ndcg']:.2f}, top-1 relevant {r['top1']:.0%}. Without the scorer, retrieval alone "
                     f"{'beats' if r['ndcg'] > base['ndcg'] else 'does not beat'} the baseline pipeline.")
    lines.append(f"- With only {total} cases, differences of a few hundredths are noise. The judgments are one person's.")
    if base["fallbacks"]:
        lines.append(f"- **{base['fallbacks']} parser answer(s) fell back to keywords**; the parser column does not measure Gemini alone.")
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval configurations on the free-text golden cases")
    parser.add_argument("--parser", choices=["keyword", "gemini"], default="keyword")
    parser.add_argument("--embedders", default="tfidf", help="comma-separated: tfidf, gemini (gemini makes live API calls)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", help="also write the report to this file")
    args = parser.parse_args(argv)

    songs = load_songs(SONGS_PATH)
    golden = load_golden()
    names = [n.strip() for n in args.embedders.split(",") if n.strip()]
    rows, notes = run_experiment(songs, golden["ranking_cases"], get_parse_fn(args.parser), names, k=args.k)
    report = format_report(rows, notes, songs, args.parser, args.k)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
