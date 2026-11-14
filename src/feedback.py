"""Interaction log and human feedback.

Every `python -m src.main --query ...` run is appended to a JSONL log with an id. A person
can then rate a run good or bad, and `export-candidates` turns the bad ones into skeleton
golden-set cases. Nothing here changes the golden set: a human must fill in which songs
would have been right and paste the case into data/golden_set.json themselves.

The log can contain personal search text, so it lives in logs/ (git-ignored).

    python -m src.feedback list
    python -m src.feedback rate <id> good|bad [--note "why"]
    python -m src.feedback export-candidates
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

LOG_PATH = "logs/interactions.jsonl"
RATINGS = ("good", "bad")


def _append(record: Dict, path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    records: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn or hand-edited line must not make the whole log unreadable
    return records


def log_interaction(
    query: str,
    parser: str,
    profile: Dict,
    notes: Sequence[str],
    recommendations: Sequence[Dict],
    path: str = LOG_PATH,
    retriever: str = "none",
    confidence: Optional[Dict] = None,
) -> str:
    """Append one run to the log and return its short id."""
    interaction_id = uuid.uuid4().hex[:8]
    _append({
        "type": "interaction",
        "id": interaction_id,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": query,
        "parser": parser,
        "retriever": retriever,
        "confidence": confidence,
        "profile": profile,
        "notes": list(notes),
        "recommendations": [
            {"id": r["id"], "title": r["title"], "score": round(r["score"], 3)} for r in recommendations
        ],
    }, path)
    return interaction_id


def record_feedback(interaction_id: str, rating: str, note: str = "", path: str = LOG_PATH) -> None:
    """Attach a good/bad rating (and optional note) to a logged interaction."""
    if rating not in RATINGS:
        raise ValueError(f"rating must be one of {RATINGS}, got {rating!r}")
    if not any(r.get("type") == "interaction" and r.get("id") == interaction_id for r in _read(path)):
        raise ValueError(f"no logged interaction with id {interaction_id!r}")
    _append({
        "type": "feedback",
        "ref": interaction_id,
        "rating": rating,
        "note": note,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, path)


def load_interactions(path: str = LOG_PATH) -> List[Dict]:
    """Interactions in log order, each with its latest rating and note (or None)."""
    records = _read(path)
    latest: Dict[str, Dict] = {}
    for r in records:
        if r.get("type") == "feedback":
            latest[r["ref"]] = r
    interactions = []
    for r in records:
        if r.get("type") == "interaction":
            fb = latest.get(r["id"])
            interactions.append({**r, "rating": fb["rating"] if fb else None, "note": fb["note"] if fb else ""})
    return interactions


def candidate_golden_cases(path: str = LOG_PATH) -> List[Dict]:
    """Skeleton golden-set cases for every query rated bad.

    ideal_ids and acceptable_ids are left empty on purpose: only a human can say which
    songs would have been right. forbidden_ids starts with the songs that were shown.
    """
    cases = []
    for r in load_interactions(path):
        if r["rating"] != "bad":
            continue
        cases.append({
            "name": f"Review {r['id']}: {r['query'][:40]}",
            "query": r["query"],
            "ideal_ids": [],
            "acceptable_ids": [],
            "forbidden_ids": [rec["id"] for rec in r["recommendations"][:3]],
            "notes": (r["note"] or "Rated bad. Fill in ideal_ids and acceptable_ids, then review forbidden_ids.")
                     + f" Shown: {', '.join(rec['title'] for rec in r['recommendations'][:3])}.",
        })
    return cases


def format_list(interactions: Sequence[Dict]) -> str:
    if not interactions:
        return "No interactions logged yet. Run `python -m src.main --query \"...\"` first.\n"
    lines = []
    for r in interactions:
        top = r["recommendations"][0]["title"] if r["recommendations"] else "(none)"
        rating = r["rating"] or "unrated"
        conf = r.get("confidence")
        tag = f" ({conf['label']} confidence)" if conf else ""
        lines.append(f"{r['id']}  {r['time']}  [{rating}]{tag}  {r['query']!r} -> {top}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interaction log and human feedback")
    parser.add_argument("--log-file", default=LOG_PATH, help="path to the JSONL log")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show logged interactions and their ratings")
    rate = sub.add_parser("rate", help="rate a logged interaction")
    rate.add_argument("id")
    rate.add_argument("rating", choices=RATINGS)
    rate.add_argument("--note", default="")
    sub.add_parser("export-candidates", help="print skeleton golden-set cases for queries rated bad")
    args = parser.parse_args(argv)

    if args.command == "list":
        print(format_list(load_interactions(args.log_file)), end="")
    elif args.command == "rate":
        try:
            record_feedback(args.id, args.rating, args.note, args.log_file)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"Recorded {args.rating!r} for {args.id}.")
    else:
        cases = candidate_golden_cases(args.log_file)
        print(json.dumps(cases, indent=2, ensure_ascii=False))
        if not cases:
            print("(no queries rated bad yet)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
