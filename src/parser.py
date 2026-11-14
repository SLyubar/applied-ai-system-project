"""Keyword-based query parser: the offline fallback for free-text requests.

This is the fallback for the Gemini-powered parser. It turns text like
"something mellow and acoustic to study to" into the same profile dict the scorer
expects. It only ever returns genres and moods that exist in the catalog.
"""

import re
from typing import Dict, Iterable, List, Tuple

from src.guardrails import normalize_label

# Words a listener might use -> mood label in the catalog.
MOOD_KEYWORDS: Dict[str, List[str]] = {
    "chill": ["chill", "calm", "mellow", "peaceful", "soothing", "laid back", "laidback", "cozy", "rainy"],
    "relaxed": ["relaxed", "relaxing", "easygoing", "easy going", "lazy", "unwind"],
    "happy": ["happy", "cheerful", "joyful", "sunny", "feel good", "feel-good", "fun", "bright"],
    "intense": ["intense", "aggressive", "hard", "powerful", "pumped"],
    "moody": ["moody", "brooding", "dark", "late night", "night drive"],
    "focused": ["focused", "focus", "concentrate", "concentration"],
    "nostalgic": ["nostalgic", "nostalgia", "throwback", "reminiscing"],
    "confident": ["confident", "swagger", "boss"],
    "melancholy": ["melancholy", "sad", "sorrowful", "gloomy", "blue", "heartbroken", "somber"],
    "euphoric": ["euphoric", "euphoria", "ecstatic", "blissful"],
    "romantic": ["romantic", "love", "date night", "sensual"],
    "angry": ["angry", "rage", "furious", "mad"],
    "hopeful": ["hopeful", "optimistic", "uplifting"],
    "energetic": ["energetic", "hype", "lively", "high energy", "high-energy"],
    "dreamy": ["dreamy", "dreaming", "floaty", "ethereal"],
}

# Words that imply a genre not literally named in the catalog.
GENRE_KEYWORDS: Dict[str, List[str]] = {
    "lofi": ["lo-fi", "lo fi", "study", "studying", "coding", "homework"],
    "hip hop": ["hiphop", "hip-hop", "rap"],
    "edm": ["electronic", "dance", "rave", "club"],
    "classical": ["orchestra", "orchestral", "symphony", "piano"],
    "rock": ["guitar"],
    "country": ["western"],
}

# Phrase -> target energy. Checked in order; the first match wins.
ENERGY_KEYWORDS: List[Tuple[float, List[str]]] = [
    (0.95, ["very high energy", "maximum energy", "full blast", "rave"]),
    # Explicit statements about energy beat cues inferred from other words.
    (0.80, ["high energy", "high-energy"]),
    (0.90, ["workout", "gym", "running", "run", "party", "intense", "pumped", "hype"]),
    (0.80, ["upbeat", "energetic", "dance", "lively"]),
    (0.15, ["sleep", "sleeping", "very quiet", "lullaby"]),
    (0.35, ["calm", "mellow", "quiet", "soft", "gentle", "slow", "relaxing", "chill", "study", "studying", "focus"]),
]
DEFAULT_ENERGY = 0.5

ACOUSTIC_YES = ["acoustic", "unplugged", "organic"]
ACOUSTIC_NO = ["electronic", "synth", "synthesized", "produced", "edm"]
_NEGATED_ACOUSTIC = re.compile(r"\b(?:not|no|non|without|avoid)[\s-]+acoustic\b")


def _contains(text: str, phrase: str) -> bool:
    """Whole-word / whole-phrase match so 'run' doesn't match 'brunch'."""
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _find_mood(text: str, known_moods: Iterable[str]) -> str:
    known = set(known_moods)
    for mood in sorted(known):  # a catalog mood named literally always wins
        if _contains(text, mood):
            return mood
    for mood, words in MOOD_KEYWORDS.items():
        if mood in known and any(_contains(text, w) for w in words):
            return mood
    return ""


def _find_genre(text: str, known_genres: Iterable[str]) -> str:
    known = set(known_genres)
    # Longest label first so "indie pop" beats "pop" and "city pop" beats "pop".
    for genre in sorted(known, key=len, reverse=True):
        if _contains(text, genre):
            return genre
    for genre, words in GENRE_KEYWORDS.items():
        if genre in known and any(_contains(text, w) for w in words):
            return genre
    return ""


def _find_energy(text: str) -> float:
    for energy, words in ENERGY_KEYWORDS:
        if any(_contains(text, w) for w in words):
            return energy
    return DEFAULT_ENERGY


def parse_query(
    query: str,
    known_genres: Iterable[str],
    known_moods: Iterable[str],
) -> Tuple[Dict, List[str]]:
    """Parse free text into (profile, notes).

    The profile uses empty strings for a genre or mood it could not identify
    (which simply never match). `likes_acoustic` is only set when the text states
    a preference. `notes` lists anything the parser had to guess.
    """
    text = normalize_label(query)
    if not text:
        raise ValueError("query is empty")

    notes: List[str] = []
    genre = _find_genre(text, known_genres)
    mood = _find_mood(text, known_moods)
    energy = _find_energy(text)

    if not genre:
        notes.append("no genre recognized; genre will not contribute to scores")
    if not mood:
        notes.append("no mood recognized; mood will not contribute to scores")
    if not any(_contains(text, w) for _, ws in ENERGY_KEYWORDS for w in ws):
        notes.append(f"no energy cue found; defaulted to {DEFAULT_ENERGY}")

    profile: Dict = {"genre": genre, "mood": mood, "energy": energy}

    if _NEGATED_ACOUSTIC.search(text):
        profile["likes_acoustic"] = False
    elif any(_contains(text, w) for w in ACOUSTIC_YES):
        profile["likes_acoustic"] = True
    elif any(_contains(text, w) for w in ACOUSTIC_NO):
        profile["likes_acoustic"] = False

    return profile, notes


def get_parse_fn(name: str):
    """Return the parser function for 'keyword' (offline) or 'gemini' (live API, falls back)."""
    if name == "keyword":
        return parse_query
    if name == "gemini":
        # Imported lazily: it needs the google-genai package and itself imports this module.
        from src.gemini_parser import make_gemini_parser

        return make_gemini_parser()
    raise ValueError(f"unknown parser {name!r}; expected 'keyword' or 'gemini'")
