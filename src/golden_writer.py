"""Write human judgments from the UI back into the golden set.

A person marks songs for a query as ideal / fine / must-not-appear, and that becomes (or
updates) a ranking case in data/golden_set.json, the same schema the evaluation reads.

Safety rules, because the golden set is the ground truth everything is measured against:
- The result must still pass the golden-set integrity tests: known song ids, no song in two
  grades, a unique name, and at least one ideal or acceptable song (a case with no
  ground truth would check nothing).
- Cases written here carry "source": "ui". A repeat save for the same query replaces that
  case. A hand-curated case for the same query is never overwritten; the save is refused.
- The file is rewritten atomically (temp file + replace), keeping its existing line endings
  and formatting, so a running evaluation never sees a half-written file and git diffs stay
  small. A running evaluation reads the golden set once at startup, so saving does not
  change a run already in progress.
"""

import json
import os
import tempfile
import threading
from typing import Dict, Iterable, List, Sequence, Tuple

from src.evaluation import GOLDEN_PATH

SOURCE = "ui"
_LOCK = threading.Lock()


def _validate(query: str, ideal: List[int], acceptable: List[int], forbidden: List[int], song_ids: set) -> None:
    if not query.strip():
        raise ValueError("query is empty")
    for label, ids in (("ideal", ideal), ("acceptable", acceptable), ("forbidden", forbidden)):
        unknown = set(ids) - song_ids
        if unknown:
            raise ValueError(f"unknown song ids in {label}: {sorted(unknown)}")
    if not (ideal or acceptable):
        raise ValueError("mark at least one song as ideal or fine; a case with no ground truth checks nothing")
    if set(ideal) & set(acceptable) or (set(ideal) | set(acceptable)) & set(forbidden):
        raise ValueError("a song cannot be in two grades")


def _unique_name(base: str, names: Iterable[str]) -> str:
    taken = set(names)
    name, n = base, 2
    while name in taken:
        name, n = f"{base} ({n})", n + 1
    return name


def _write_atomic(path: str, raw: str, text: str) -> None:
    crlf = "\r\n" in raw
    if crlf:
        text = text.replace("\n", "\r\n")
    if raw.endswith("\n"):
        text += "\r\n" if crlf else "\n"
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_case(
    query: str,
    ideal_ids: Sequence[int],
    acceptable_ids: Sequence[int],
    forbidden_ids: Sequence[int],
    songs: List[Dict],
    notes: str = "",
    path: str = GOLDEN_PATH,
) -> Tuple[Dict, str]:
    """Add or update the UI case for `query`. Returns (case, "added" | "updated")."""
    query = query.strip()
    ideal, acceptable, forbidden = list(dict.fromkeys(ideal_ids)), list(dict.fromkeys(acceptable_ids)), list(dict.fromkeys(forbidden_ids))
    _validate(query, ideal, acceptable, forbidden, {s["id"] for s in songs})

    with _LOCK:
        with open(path, encoding="utf-8", newline="") as f:
            raw = f.read()
        golden = json.loads(raw)
        cases = golden["ranking_cases"]
        existing = [c for c in cases if c.get("query", "").strip().lower() == query.lower()]
        mine = next((c for c in existing if c.get("source") == SOURCE), None)
        if existing and mine is None:
            raise ValueError(
                f"the golden set already has a hand-curated case for this query ({existing[0]['name']!r}); "
                "edit data/golden_set.json directly to change it"
            )
        fields = {
            "ideal_ids": ideal, "acceptable_ids": acceptable, "forbidden_ids": forbidden,
            "notes": notes.strip() or "Added from the Streamlit app.",
        }
        if mine is not None:
            mine.update(fields)
            case, action = mine, "updated"
        else:
            case = {
                "name": _unique_name(f"UI: {query[:40]}", (c["name"] for c in cases)),
                "query": query, **fields, "source": SOURCE,
            }
            cases.append(case)
            action = "added"
        _write_atomic(path, raw, json.dumps(golden, indent=2, ensure_ascii=False))
    return case, action


def find_ui_case(query: str, path: str = GOLDEN_PATH) -> Dict:
    """The UI-written case for `query`, or {} (used to pre-fill the form on a repeat query)."""
    try:
        with open(path, encoding="utf-8") as f:
            cases = json.load(f)["ranking_cases"]
    except (OSError, ValueError, KeyError):
        return {}
    return next(
        (c for c in cases if c.get("source") == SOURCE and c.get("query", "").strip().lower() == query.strip().lower()),
        {},
    )
