"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import argparse

from src.confidence import assess
from src.feedback import LOG_PATH, log_interaction
from src.guardrails import known_labels, sanitize_profile
from src.parser import get_parse_fn
from src.recommender import load_songs, recommend_songs
from src.retrieval import Retriever, get_embedder

DEFAULT_POOL = 20  # candidate songs retrieved before scoring


PROFILES = {
    # Standard profiles
    "Starter lofi": {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True},
    "Pop workout": {"genre": "pop", "mood": "chill", "energy": 0.9, "likes_acoustic": False},
    "Rock intense": {"genre": "rock", "mood": "intense", "energy": 0.7, "likes_acoustic": True},

    # Adversarial / edge-case profiles
    "Conflict: high-energy sad classical": {
        "genre": "classical", "mood": "sad", "energy": 0.9, "likes_acoustic": True,
    },
    "Unknown genre and mood": {
        "genre": "vaporwave", "mood": "sad", "energy": 0.5, "likes_acoustic": False,
    },
    "Out-of-range energy (1.5)": {
        "genre": "pop", "mood": "happy", "energy": 1.5, "likes_acoustic": False,
    },
    "No matches, no acoustic pref": {"genre": "polka", "mood": "zen", "energy": 0.5},
    "Metal fan wants calm": {
        "genre": "metal", "mood": "chill", "energy": 0.1, "likes_acoustic": True,
    },
    "Messy casing and whitespace": {
        "genre": "Lofi ", "mood": "CHILL", "energy": 0.4, "likes_acoustic": True,
    },
    "Acoustic lover wants EDM": {
        "genre": "edm", "mood": "euphoric", "energy": 0.96, "likes_acoustic": True,
    },
}


def describe_acoustic(user_prefs: dict) -> str:
    if "likes_acoustic" not in user_prefs:
        return "no acoustic preference"
    return "likes acoustic" if user_prefs["likes_acoustic"] else "prefers non-acoustic"


def print_recommendations(
    name: str, user_prefs: dict, songs: list, k: int = 5, notes: list = None, pool: list = None
) -> tuple:
    """Print the top k songs with a confidence assessment.

    Returns (cleaned profile, [{id, title, score}, ...], Confidence) so callers can log the run.
    `pool` limits ranking to a subset of `songs` (the retrieved candidates); labels are still checked
    against the whole catalog.
    """
    genres, moods = known_labels(songs)
    raw_prefs = user_prefs  # the parser's own output, including its reported certainty
    user_prefs, warnings = sanitize_profile(user_prefs, genres, moods)
    recommendations = recommend_songs(user_prefs, pool if pool is not None else songs, k=k)
    confidence = assess(
        raw_prefs, notes or [], genres, moods, recommendations[0][1] if recommendations else None
    )
    divider = "-" * 60

    print("\n" + "=" * 60)
    print(f"PROFILE: {name}")
    print(
        f"{user_prefs['genre']!r} | {user_prefs['mood']!r} | "
        f"energy {user_prefs['energy']:.2f} | {describe_acoustic(user_prefs)}"
    )
    print("=" * 60)
    for message in (notes or []) + warnings:
        print(f"  ! {message}")
    print(
        f"  Confidence: {confidence.label.upper()} ({confidence.score:.2f})  "
        f"[understanding {confidence.understanding:.2f}, match {confidence.match:.2f}]  "
        f"matched on: {', '.join(confidence.criteria) or 'nothing'}"
    )
    for reason in confidence.reasons:
        print(f"    - {reason}")
    if confidence.label == "low":
        print("  Low confidence: treat these as rough suggestions, not a good match.")

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n{rank}. {song['title']} - {song['artist']}")
        print(f"   Score: {score:.2f} / 1.00")
        print("   Why:")
        for reason in explanation.split("; "):
            print(f"     * {reason}")
        print(divider)

    return (
        user_prefs,
        [{"id": song["id"], "title": song["title"], "score": score} for song, score, _ in recommendations],
        confidence,
    )


def main(argv=None) -> None:
    arg_parser = argparse.ArgumentParser(description="Music Recommender")
    arg_parser.add_argument(
        "--query",
        help='free-text request, e.g. "something mellow and acoustic to study to"',
    )
    arg_parser.add_argument(
        "--parser", choices=["keyword", "gemini"], default="keyword",
        help="how --query is parsed; 'gemini' calls the API and falls back to keywords on failure",
    )
    arg_parser.add_argument(
        "--retriever", choices=["none", "tfidf", "gemini"], default="none",
        help="retrieve candidate songs by description before scoring; 'gemini' uses embeddings (live API, cached)",
    )
    arg_parser.add_argument(
        "--pool", type=int, default=DEFAULT_POOL,
        help="how many candidates the retriever passes to the scorer (default: %(default)s)",
    )
    arg_parser.add_argument("--no-log", action="store_true", help="do not record this --query run in the interaction log")
    arg_parser.add_argument("--log-file", default=LOG_PATH, help="where --query runs are logged (default: %(default)s)")
    args = arg_parser.parse_args(argv)

    songs = load_songs("data/songs.csv")

    if args.query:
        genres, moods = known_labels(songs)
        user_prefs, notes = get_parse_fn(args.parser)(args.query, genres, moods)
        pool = None
        retriever_used = "none"
        if args.retriever != "none":
            try:
                retriever = Retriever(songs, get_embedder(args.retriever))
                found = retriever.search(args.query, max(args.pool, 5))
                keep = {song_id for song_id, _ in found}
                pool = [s for s in songs if s["id"] in keep]
                retriever_used = args.retriever
                titles = {s["id"]: s["title"] for s in songs}
                notes = list(notes) + [
                    f"retrieved {len(pool)} candidates ({args.retriever}); closest: "
                    + ", ".join(titles[song_id] for song_id, _ in found[:5])
                ]
            except Exception as error:  # no key, quota or network: say so and rank the whole catalog
                notes = list(notes) + [
                    f"retrieval unavailable ({type(error).__name__}: {' '.join(str(error).split())[:100]}); "
                    "ranked the whole catalog"
                ]
        clean_prefs, recommendations, confidence = print_recommendations(
            f"Query: {args.query}", user_prefs, songs, notes=notes, pool=pool
        )
        if not args.no_log:
            interaction_id = log_interaction(
                args.query, args.parser, clean_prefs, notes, recommendations, args.log_file, retriever_used,
                confidence.as_dict(),
            )
            print(f"\nLogged as {interaction_id}. Rate it with:  python -m src.feedback rate {interaction_id} good|bad")
        return

    for name, user_prefs in PROFILES.items():
        print_recommendations(name, user_prefs, songs)


if __name__ == "__main__":
    main()
