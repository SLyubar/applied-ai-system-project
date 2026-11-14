"""Confidence scoring: how much should anyone trust a set of recommendations?

Two simple, explainable numbers, combined by taking the weaker one:

- understanding: of the criteria the request specified (energy, mood, genre, acoustic),
  what share could be mapped to something the catalog knows, weighted like the scorer.
  A genre the catalog has never heard of, or a request with no usable cue at all, lowers it.
  An LLM parser's own reported certainty can lower it further but never raise it.
- match: how well the best song fits the criteria that were understood.

    confidence = min(understanding, match)

It is deliberately not a probability. It says "the system had a lot (or very little) to go
on, and the best song fits (or barely fits) it", and always lists the reasons. How well it
separates good results from bad ones is measured in the evaluation report, not assumed.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.guardrails import normalize_label
from src.recommender import WEIGHTS

HIGH = 0.8    # at or above: high confidence
MEDIUM = 0.6  # at or above: medium; below: low
LOW_PARSER_CERTAINTY = 0.6  # a parser reporting less than this is called out in the reasons

NO_ENERGY_NOTE = "no energy cue"
# Parsers drop labels the catalog does not have and say so; that still means the user asked for one.
_DROPPED_LABEL = re.compile(r"returned (genre|mood) .* not in the catalog; ignored")


@dataclass
class Confidence:
    score: float
    label: str
    understanding: float
    match: float
    criteria: List[str] = field(default_factory=list)  # what the results were actually matched on
    reasons: List[str] = field(default_factory=list)   # why it is not high, empty when it is

    def as_dict(self) -> Dict:
        return {
            "score": round(self.score, 3), "label": self.label, "understanding": round(self.understanding, 3),
            "match": round(self.match, 3), "criteria": list(self.criteria), "reasons": list(self.reasons),
        }


def label_for(score: float) -> str:
    return "high" if score >= HIGH else "medium" if score >= MEDIUM else "low"


def analyze_request(
    profile: Dict, notes: Sequence[str], genres: Iterable[str], moods: Iterable[str]
) -> Tuple[Set[str], Set[str], List[str]]:
    """Return (specified features, understood features, reasons something was not understood)."""
    genres, moods = set(genres), set(moods)
    specified: Set[str] = set()
    understood: Set[str] = set()
    reasons: List[str] = []

    if not any(NO_ENERGY_NOTE in n for n in notes) and profile.get("energy") is not None:
        specified.add("energy")
        understood.add("energy")

    dropped = {m.group(1) for n in notes for m in [_DROPPED_LABEL.search(n)] if m}
    for feature, known in (("genre", genres), ("mood", moods)):
        label = normalize_label(profile.get(feature) or "")
        if label:
            specified.add(feature)
            if label in known:
                understood.add(feature)
            else:
                reasons.append(f"{feature} {label!r} is not in the catalog")
        elif feature in dropped:
            specified.add(feature)
            reasons.append(f"a {feature} in the request is not in the catalog")

    if profile.get("likes_acoustic") is not None:
        specified.add("acoustic")
        understood.add("acoustic")
    return specified, understood, reasons


def _weight(features: Iterable[str]) -> float:
    return sum(WEIGHTS[f] for f in features)


def understanding(
    profile: Dict, notes: Sequence[str], genres: Iterable[str], moods: Iterable[str]
) -> Tuple[float, List[str]]:
    """How much of what the request specified could be understood (0 to 1), plus the reasons it fell short."""
    specified, understood, reasons = analyze_request(profile, notes, genres, moods)
    if not specified:
        return 0.0, ["nothing in the request could be turned into a criterion"]
    score = _weight(understood) / _weight(specified)

    reported = profile.get("parser_confidence")
    if isinstance(reported, (int, float)) and not isinstance(reported, bool):
        reported = min(1.0, max(0.0, float(reported)))
        if reported < LOW_PARSER_CERTAINTY:
            reasons.append(f"the parser reported low certainty ({reported:.2f})")
        score = min(score, reported)
    return score, reasons


def assess(
    profile: Dict,
    notes: Sequence[str],
    genres: Iterable[str],
    moods: Iterable[str],
    top_score: Optional[float],
) -> Confidence:
    """Confidence in a recommendation list whose best song scored `top_score` against the profile."""
    specified, understood, _ = analyze_request(profile, notes, genres, moods)
    u_score, reasons = understanding(profile, notes, genres, moods)

    achievable = _weight(understood)
    if top_score is None or achievable == 0:
        match = 0.0
    else:
        match = min(1.0, max(0.0, top_score) / achievable)
    if achievable and top_score is not None and match < HIGH:
        reasons.append(f"the best song fits only {match:.0%} of what was understood")
    if top_score is None:
        reasons.append("there were no songs to recommend")

    score = min(u_score, match)
    order = ["energy", "mood", "genre", "acoustic"]
    return Confidence(
        score=score, label=label_for(score), understanding=u_score, match=match,
        criteria=[f for f in order if f in understood],
        reasons=[] if label_for(score) == "high" and not reasons else reasons,
    )


def calibration_table(rows: Sequence[Tuple[str, bool]]) -> List[Dict]:
    """Group (confidence label, passed) pairs into per-band counts and pass rates.

    This is the check that the score means something: pass rates should fall from high to low.
    """
    table = []
    for band in ("high", "medium", "low"):
        results = [passed for label, passed in rows if label == band]
        table.append({
            "band": band, "cases": len(results), "passed": sum(1 for p in results if p),
            "pass_rate": (sum(1 for p in results if p) / len(results)) if results else None,
        })
    return table
