"""One free-text request -> parsed profile, ranked songs and a confidence assessment.

This is the same flow `python -m src.main --query ...` runs, returned as data instead of
printed, so the Streamlit app (and anything else) can reuse it. The CLI is unchanged.
"""

from typing import Callable, Dict, List, Optional

from src.confidence import assess
from src.guardrails import known_labels, sanitize_profile
from src.parser import get_parse_fn
from src.recommender import recommend_songs
from src.retrieval import Retriever, get_embedder

PARSERS = ("keyword", "gemini")
RETRIEVERS = ("none", "tfidf", "gemini")


def _short(error: Exception) -> str:
    return f"{type(error).__name__}: {' '.join(str(error).split())[:100]}"


def run_query(
    query: str,
    songs: List[Dict],
    parser: str = "keyword",
    retriever: str = "none",
    pool: int = 20,
    k: int = 5,
    parse_fn: Optional[Callable] = None,
) -> Dict:
    """Parse `query`, optionally retrieve candidates, rank, and assess confidence.

    Never raises for a failing live service: an unavailable Gemini parser or retriever is
    reported in `notes` and the run falls back to keywords / the whole catalog.
    Raises ValueError only for an empty query.
    """
    if not query or not query.strip():
        raise ValueError("query is empty")
    genres, moods = known_labels(songs)

    notes: List[str] = []
    parser_used = parser
    if parse_fn is None:
        try:
            parse_fn = get_parse_fn(parser)
        except Exception as error:  # e.g. no API key or missing package
            notes.append(f"{parser} parser unavailable ({_short(error)}); used the keyword parser")
            parser_used = "keyword"
            parse_fn = get_parse_fn("keyword")
    raw_prefs, parse_notes = parse_fn(query, genres, moods)
    notes += list(parse_notes)

    candidates = None
    retriever_used = "none"
    if retriever != "none":
        try:
            found = Retriever(songs, get_embedder(retriever)).search(query, max(pool, 5))
            keep = {song_id for song_id, _ in found}
            candidates = [s for s in songs if s["id"] in keep]
            retriever_used = retriever
            titles = {s["id"]: s["title"] for s in songs}
            notes.append(
                f"retrieved {len(candidates)} candidates ({retriever}); closest: "
                + ", ".join(titles[song_id] for song_id, _ in found[:5])
            )
        except Exception as error:
            notes.append(f"retrieval unavailable ({_short(error)}); ranked the whole catalog")

    profile, warnings = sanitize_profile(raw_prefs, genres, moods)
    ranked = recommend_songs(profile, candidates if candidates is not None else songs, k=k)
    confidence = assess(raw_prefs, notes, genres, moods, ranked[0][1] if ranked else None)
    return {
        "query": query.strip(),
        "parser": parser_used,
        "retriever": retriever_used,
        "profile": profile,
        "notes": notes + warnings,
        "recommendations": [
            {"id": song["id"], "title": song["title"], "artist": song["artist"], "score": score,
             "reasons": explanation.split("; ")}
            for song, score, explanation in ranked
        ],
        "confidence": confidence,
    }
