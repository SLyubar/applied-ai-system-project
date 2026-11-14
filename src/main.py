"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import argparse

from src.guardrails import known_labels, sanitize_profile
from src.parser import get_parse_fn
from src.recommender import load_songs, recommend_songs


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
    name: str, user_prefs: dict, songs: list, k: int = 5, notes: list = None
) -> None:
    genres, moods = known_labels(songs)
    user_prefs, warnings = sanitize_profile(user_prefs, genres, moods)
    recommendations = recommend_songs(user_prefs, songs, k=k)
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

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n{rank}. {song['title']} - {song['artist']}")
        print(f"   Score: {score:.2f} / 1.00")
        print("   Why:")
        for reason in explanation.split("; "):
            print(f"     * {reason}")
        print(divider)


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Music Recommender")
    arg_parser.add_argument(
        "--query",
        help='free-text request, e.g. "something mellow and acoustic to study to"',
    )
    arg_parser.add_argument(
        "--parser", choices=["keyword", "gemini"], default="keyword",
        help="how --query is parsed; 'gemini' calls the API and falls back to keywords on failure",
    )
    args = arg_parser.parse_args()

    songs = load_songs("data/songs.csv")

    if args.query:
        genres, moods = known_labels(songs)
        user_prefs, notes = get_parse_fn(args.parser)(args.query, genres, moods)
        print_recommendations(f"Query: {args.query}", user_prefs, songs, notes=notes)
        return

    for name, user_prefs in PROFILES.items():
        print_recommendations(name, user_prefs, songs)


if __name__ == "__main__":
    main()
