"""Input guardrails: clean and validate a user profile before it reaches the scorer."""

import re
from typing import Dict, Iterable, List, Optional, Tuple

_TRUE_STRINGS = {"true", "yes", "y", "1"}
_FALSE_STRINGS = {"false", "no", "n", "0"}


def normalize_label(value) -> str:
    """Lowercase, trim, and collapse internal whitespace (e.g. ' Hip   Hop ' -> 'hip hop')."""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def known_labels(songs: List[Dict]) -> Tuple[List[str], List[str]]:
    """Return the sorted genres and moods present in the catalog."""
    genres = sorted({normalize_label(s["genre"]) for s in songs})
    moods = sorted({normalize_label(s["mood"]) for s in songs})
    return genres, moods


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_label(value)
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    raise ValueError(f"likes_acoustic must be a boolean, got {value!r}")


def sanitize_profile(
    prefs: Dict,
    known_genres: Optional[Iterable[str]] = None,
    known_moods: Optional[Iterable[str]] = None,
) -> Tuple[Dict, List[str]]:
    """Return (clean_prefs, warnings) for a raw profile dict.

    - genre and mood are normalized (trimmed, lowercased); a label missing from the
      catalog produces a warning because it can never match any song.
    - energy is clamped to 0.0-1.0 with a warning if it was out of range.
    - likes_acoustic is optional; when absent the acoustic term is skipped.

    Raises ValueError if genre, mood, or energy is missing or unusable.
    """
    warnings: List[str] = []

    for key in ("genre", "mood", "energy"):
        if key not in prefs or prefs[key] is None:
            raise ValueError(f"profile is missing required field {key!r}")

    clean: Dict = {
        "genre": normalize_label(prefs["genre"]),
        "mood": normalize_label(prefs["mood"]),
    }

    if str(prefs["genre"]) != clean["genre"]:
        warnings.append(f"genre {prefs['genre']!r} normalized to {clean['genre']!r}")
    if str(prefs["mood"]) != clean["mood"]:
        warnings.append(f"mood {prefs['mood']!r} normalized to {clean['mood']!r}")

    try:
        energy = float(prefs["energy"])
    except (TypeError, ValueError):
        raise ValueError(f"energy must be a number, got {prefs['energy']!r}") from None
    if energy != energy:  # NaN
        raise ValueError("energy must be a number, got NaN")
    clamped = min(1.0, max(0.0, energy))
    if clamped != energy:
        warnings.append(f"energy {energy} is out of range and was clamped to {clamped}")
    clean["energy"] = clamped

    if "likes_acoustic" in prefs and prefs["likes_acoustic"] is not None:
        clean["likes_acoustic"] = _coerce_bool(prefs["likes_acoustic"])

    if known_genres is not None and clean["genre"] not in set(known_genres):
        warnings.append(f"genre {clean['genre']!r} is not in the catalog, so it cannot match any song")
    if known_moods is not None and clean["mood"] not in set(known_moods):
        warnings.append(f"mood {clean['mood']!r} is not in the catalog, so it cannot match any song")

    return clean, warnings
