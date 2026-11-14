"""Robustness checks for a query parser: paraphrase consistency and adversarial input.

Both take any parse_fn(query, genres, moods) -> (profile, notes), so they work for the
keyword parser and the Gemini parser alike. They are run by `python -m src.evaluation`
and reported in the same file.

- Paraphrase groups: different wordings of one intent should give a compatible profile.
  A group passes only if every phrasing meets the group's expectations and the parsed
  energies agree within MAX_ENERGY_SPREAD.
- Adversarial cases: prompt injection, other languages, gibberish, off-topic text, very
  long input. A case passes only if the profile is safe (only catalog labels, energy in
  range) and meets any stated expectations.
"""

from typing import Callable, Dict, List, Sequence, Tuple

from src.confidence import understanding
from src.evaluation import FALLBACK_NOTE_PREFIX, check_parse
from src.guardrails import known_labels

MAX_ENERGY_SPREAD = 0.25  # paraphrases of one intent may differ by at most this much in energy

ParseFn = Callable[[str, Sequence[str], Sequence[str]], Tuple[Dict, List[str]]]


def check_safe(profile: Dict, genres: Sequence[str], moods: Sequence[str]) -> List[str]:
    """Return the problems that make a parsed profile unsafe to hand to the scorer."""
    problems: List[str] = []
    if profile.get("genre", "") not in set(genres) | {""}:
        problems.append(f"genre {profile.get('genre')!r} is not a catalog label")
    if profile.get("mood", "") not in set(moods) | {""}:
        problems.append(f"mood {profile.get('mood')!r} is not a catalog label")
    energy = profile.get("energy")
    if not isinstance(energy, (int, float)) or isinstance(energy, bool) or not 0.0 <= energy <= 1.0:
        problems.append(f"energy {energy!r} is not a number between 0 and 1")
    if "likes_acoustic" in profile and not isinstance(profile["likes_acoustic"], bool):
        problems.append("likes_acoustic is not a boolean")
    return problems


def _parse(parse_fn: ParseFn, query: str, genres, moods) -> Tuple[Dict, bool, str, List[str]]:
    """Parse one query. Returns (profile, fell_back, error, notes); error is '' on success."""
    try:
        profile, notes = parse_fn(query, genres, moods)
    except ValueError as error:
        return {}, False, str(error), []
    return profile, any(n.startswith(FALLBACK_NOTE_PREFIX) for n in notes), "", notes


def evaluate_paraphrases(groups: List[Dict], songs: List[Dict], parse_fn: ParseFn) -> List[Dict]:
    genres, moods = known_labels(songs)
    results: List[Dict] = []
    for group in groups:
        energies: List[float] = []
        phrase_failures: List[str] = []
        fallbacks = 0
        for phrase in group["queries"]:
            profile, fell_back, error, _ = _parse(parse_fn, phrase, genres, moods)
            fallbacks += 1 if fell_back else 0
            failed = check_parse(profile, group["expected"]) if not error else [f"error: {error}"]
            if failed:
                phrase_failures.append(f"{phrase!r}: {', '.join(failed)}")
            if isinstance(profile.get("energy"), (int, float)):
                energies.append(float(profile["energy"]))
        spread = max(energies) - min(energies) if len(energies) >= 2 else 0.0
        results.append({
            "name": group["name"],
            "phrases": len(group["queries"]),
            "phrases_passed": len(group["queries"]) - len(phrase_failures),
            "failures": phrase_failures,
            "energy_spread": spread,
            "passed": not phrase_failures and spread <= MAX_ENERGY_SPREAD,
            "fallbacks": fallbacks,
        })
    return results


def evaluate_adversarial(cases: List[Dict], songs: List[Dict], parse_fn: ParseFn) -> List[Dict]:
    genres, moods = known_labels(songs)
    results: List[Dict] = []
    for case in cases:
        query = case["query"] * case.get("query_repeat", 1)
        profile, fell_back, error, notes = _parse(parse_fn, query, genres, moods)
        u_score = None
        if error:
            problems, failed = [], [f"error: {error}"]
        else:
            problems = check_safe(profile, genres, moods)
            expected = case.get("expected", {})
            failed = check_parse(profile, expected)
            # Junk input must be rated as poorly understood; clear input must not be.
            u_score = understanding(profile, notes, genres, moods)[0]
            if "understanding_max" in expected and u_score > expected["understanding_max"]:
                failed.append("understanding_max")
            if "understanding_min" in expected and u_score < expected["understanding_min"]:
                failed.append("understanding_min")
        results.append({
            "name": case["name"],
            "query": query,
            "problems": problems,
            "failed_fields": failed,
            "understanding": u_score,
            "passed": not problems and not failed,
            "fell_back": fell_back,
        })
    return results


def summarize_robustness(paraphrase: List[Dict], adversarial: List[Dict]) -> Dict:
    return {
        "paraphrase_passed": sum(1 for r in paraphrase if r["passed"]),
        "paraphrase_total": len(paraphrase),
        "adversarial_passed": sum(1 for r in adversarial if r["passed"]),
        "adversarial_total": len(adversarial),
        "fallbacks": sum(r["fallbacks"] for r in paraphrase) + sum(1 for r in adversarial if r["fell_back"]),
        "calls": sum(r["phrases"] for r in paraphrase) + len(adversarial),
    }


def format_section(runs: List[Dict]) -> str:
    """Markdown for the robustness results of one or more runs.

    Each run is {"paraphrase": [...], "adversarial": [...]} as returned by the
    evaluate_* functions. Cases are listed with how many runs they passed.
    """
    n = len(runs)
    summaries = [summarize_robustness(r["paraphrase"], r["adversarial"]) for r in runs]
    fallbacks = sum(s["fallbacks"] for s in summaries)
    calls = sum(s["calls"] for s in summaries)

    def passes(kind: str, name: str) -> int:
        return sum(1 for run in runs for r in run[kind] if r["name"] == name and r["passed"])

    lines = [
        "## Robustness",
        "",
        f"Runs: {n}. A paraphrase group passes only if every wording meets the expectation and the energies "
        f"agree within {MAX_ENERGY_SPREAD}. An adversarial case passes only if the profile is safe "
        "(catalog labels only, energy in range) and meets any stated expectation.",
        "",
        f"| Metric | Value |",
        "|---|---|",
        f"| Paraphrase groups passed (mean) | "
        f"{sum(s['paraphrase_passed'] for s in summaries) / n:.1f} of {summaries[0]['paraphrase_total']} |",
        f"| Adversarial cases passed (mean) | "
        f"{sum(s['adversarial_passed'] for s in summaries) / n:.1f} of {summaries[0]['adversarial_total']} |",
        f"| Answers that fell back to keywords | {fallbacks}/{calls} |",
        "",
        "### Paraphrase groups",
        "",
        "| Group | Passed | Phrasings passed (last run) | Worst energy spread |",
        "|---|---|---|---|",
    ]
    for r in runs[-1]["paraphrase"]:
        worst = max(x["energy_spread"] for run in runs for x in run["paraphrase"] if x["name"] == r["name"])
        lines.append(
            f"| {r['name']} | {passes('paraphrase', r['name'])}/{n} | {r['phrases_passed']}/{r['phrases']} | {worst:.2f} |"
        )
    lines += ["", "### Adversarial and edge-case queries", "", "| Case | Passed | Why it failed (last run) |", "|---|---|---|"]
    for r in runs[-1]["adversarial"]:
        why = "; ".join(r["problems"] + r["failed_fields"]) or "-"
        lines.append(f"| {r['name']} | {passes('adversarial', r['name'])}/{n} | {why} |")

    failing = [r["name"] for r in runs[-1]["paraphrase"] if passes("paraphrase", r["name"]) < n]
    failing += [r["name"] for r in runs[-1]["adversarial"] if passes("adversarial", r["name"]) < n]
    lines += ["", "Review: " + (("did not pass every run: " + "; ".join(failing) + ".") if failing else "every case passed every run.")]
    if fallbacks:
        lines.append(
            f"\n**{fallbacks} robustness answer(s) came from the keyword fallback, not Gemini**, "
            "so these numbers do not measure Gemini alone."
        )
    return "\n".join(lines) + "\n"
