import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict, dataclass

from src.guardrails import sanitize_profile


def _profile_to_prefs(user: "UserProfile") -> Dict:
    """Convert a UserProfile to the dict the scorer expects, cleaning labels and energy."""
    prefs, _warnings = sanitize_profile(
        {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
    )
    return prefs


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    description: str = ""  # free text used by the retriever

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs for this user, highest score first.

        Uses the same scoring rule as recommend_songs(). Ties keep catalog order.
        """
        if k <= 0:
            return []
        prefs = _profile_to_prefs(user)
        scored = [(song, score_song(prefs, asdict(song))[0]) for song in self.songs]
        scored.sort(key=lambda item: item[1], reverse=True)  # sort is stable
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain a song's score as '<score> / 1.00: reason; reason; ...'."""
        score, reasons = score_song(_profile_to_prefs(user), asdict(song))
        return f"{score:.2f} / 1.00: " + "; ".join(reasons)

WEIGHTS = {"energy": 0.40, "mood": 0.25, "genre": 0.20, "acoustic": 0.15}

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file as dicts with numeric fields converted to int/float."""
    print(f"Loading songs from {csv_path}...")
    int_fields = ("id",)
    float_fields = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")

    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)
    
    print(f"Loaded songs: {len(songs)}")
    
    return songs

def score_song(user_prefs: Dict, song: Dict, weights: Optional[Dict[str, float]] = None) -> Tuple[float, List[str]]:
    """Score one song (max 1.0 with the default weights) against user preferences and return (score, reasons).

    `weights` overrides WEIGHTS; used by the ablation study.
    """
    w = WEIGHTS if weights is None else weights
    score = 0.0
    reasons: List[str] = []

    energy_score = 1 - abs(song["energy"] - user_prefs["energy"])
    points = w["energy"] * energy_score
    score += points
    reasons.append(
        f"energy {song['energy']:.2f} vs target {user_prefs['energy']:.2f} (+{points:.2f})"
    )

    if song["mood"].lower() == user_prefs["mood"].lower():
        score += w["mood"]
        reasons.append(f"mood match (+{w['mood']:.2f})")

    if song["genre"].lower() == user_prefs["genre"].lower():
        score += w["genre"]
        reasons.append(f"genre match (+{w['genre']:.2f})")

    # No stated acoustic preference: skip the term rather than assume one.
    if "likes_acoustic" in user_prefs:
        acoustic_score = (
            song["acousticness"] if user_prefs["likes_acoustic"] else 1 - song["acousticness"]
        )
        points = w["acoustic"] * acoustic_score
        score += points
        reasons.append(f"acoustic fit {acoustic_score:.2f} (+{points:.2f})")

    return score, reasons

def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5, weights: Optional[Dict[str, float]] = None
) -> List[Tuple[Dict, float, str]]:
    """Return the top k (song, score, explanation) tuples, highest score first."""
    scored = [
        (song, score, "; ".join(reasons))
        for song in songs
        for score, reasons in [score_song(user_prefs, song, weights)]
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
