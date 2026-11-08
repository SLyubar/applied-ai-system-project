import pytest

from src.guardrails import known_labels, normalize_label, sanitize_profile
from src.recommender import load_songs, recommend_songs

GENRES = ["lofi", "pop"]
MOODS = ["chill", "happy"]


def test_normalize_label_trims_lowercases_and_collapses_spaces():
    assert normalize_label("  Hip   Hop ") == "hip hop"


def test_messy_labels_are_normalized_and_reported():
    prefs = {"genre": "Lofi ", "mood": "CHILL", "energy": 0.4, "likes_acoustic": True}
    clean, warnings = sanitize_profile(prefs, GENRES, MOODS)
    assert clean["genre"] == "lofi"
    assert clean["mood"] == "chill"
    assert len(warnings) == 2


def test_out_of_range_energy_is_clamped_with_warning():
    high, warn_high = sanitize_profile({"genre": "pop", "mood": "happy", "energy": 1.5})
    low, warn_low = sanitize_profile({"genre": "pop", "mood": "happy", "energy": -2})
    assert high["energy"] == 1.0
    assert low["energy"] == 0.0
    assert any("clamped" in w for w in warn_high + warn_low)


def test_in_range_energy_has_no_warning():
    clean, warnings = sanitize_profile({"genre": "pop", "mood": "happy", "energy": 0.5}, GENRES, MOODS)
    assert clean["energy"] == 0.5
    assert warnings == []


def test_unknown_labels_are_flagged():
    _, warnings = sanitize_profile({"genre": "vaporwave", "mood": "zen", "energy": 0.5}, GENRES, MOODS)
    assert len(warnings) == 2
    assert all("not in the catalog" in w for w in warnings)


def test_acoustic_preference_is_optional_and_coerced():
    absent, _ = sanitize_profile({"genre": "pop", "mood": "happy", "energy": 0.5})
    assert "likes_acoustic" not in absent
    coerced, _ = sanitize_profile(
        {"genre": "pop", "mood": "happy", "energy": 0.5, "likes_acoustic": "Yes"}
    )
    assert coerced["likes_acoustic"] is True


@pytest.mark.parametrize(
    "bad",
    [
        {"mood": "happy", "energy": 0.5},
        {"genre": "pop", "energy": 0.5},
        {"genre": "pop", "mood": "happy"},
        {"genre": "pop", "mood": "happy", "energy": "loud"},
        {"genre": "pop", "mood": "happy", "energy": float("nan")},
        {"genre": "pop", "mood": "happy", "energy": 0.5, "likes_acoustic": "maybe"},
    ],
)
def test_unusable_profiles_raise_value_error(bad):
    with pytest.raises(ValueError):
        sanitize_profile(bad)


def test_known_labels_come_from_catalog():
    songs = load_songs("data/songs.csv")
    genres, moods = known_labels(songs)
    assert "lofi" in genres and "indie pop" in genres
    assert "chill" in moods and "melancholy" in moods


def test_sanitizing_restores_the_genre_match_lost_to_whitespace():
    """Regression: 'Lofi ' used to score 0.76 because the trailing space broke the genre match."""
    songs = load_songs("data/songs.csv")
    raw = {"genre": "Lofi ", "mood": "CHILL", "energy": 0.4, "likes_acoustic": True}
    clean, _ = sanitize_profile(raw)

    before = recommend_songs(raw, songs, k=1)[0][1]
    after = recommend_songs(clean, songs, k=1)[0][1]
    assert after == pytest.approx(0.96, abs=0.005)
    assert after > before
