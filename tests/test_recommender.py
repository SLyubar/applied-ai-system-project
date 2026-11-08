import pytest

from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    load_songs,
    recommend_songs,
    score_song,
)
from dataclasses import asdict


def make_song(id, title, genre="pop", mood="happy", energy=0.5, acousticness=0.5):
    return Song(
        id=id, title=title, artist="Test Artist", genre=genre, mood=mood,
        energy=energy, tempo_bpm=100, valence=0.5, danceability=0.5,
        acousticness=acousticness,
    )


POP_USER = UserProfile(
    favorite_genre="pop", favorite_mood="happy", target_energy=0.8, likes_acoustic=False
)

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---- Recommender.recommend ----

def test_recommend_orders_full_ranking_best_first():
    songs = [
        make_song(1, "Miss", genre="jazz", mood="sad", energy=0.1, acousticness=0.9),
        make_song(2, "Perfect", genre="pop", mood="happy", energy=0.8, acousticness=0.0),
        make_song(3, "Partial", genre="pop", mood="sad", energy=0.7, acousticness=0.2),
    ]
    results = Recommender(songs).recommend(POP_USER, k=3)
    assert [s.title for s in results] == ["Perfect", "Partial", "Miss"]


def test_recommend_respects_k():
    rec = Recommender([make_song(i, f"S{i}", energy=i / 10) for i in range(1, 6)])
    assert len(rec.recommend(POP_USER, k=3)) == 3
    assert len(rec.recommend(POP_USER, k=1)) == 1


def test_recommend_k_larger_than_catalog_returns_everything():
    rec = Recommender([make_song(1, "A"), make_song(2, "B")])
    assert len(rec.recommend(POP_USER, k=50)) == 2


@pytest.mark.parametrize("k", [0, -1])
def test_recommend_non_positive_k_returns_empty(k):
    assert Recommender([make_song(1, "A")]).recommend(POP_USER, k=k) == []


def test_recommend_empty_catalog_returns_empty():
    assert Recommender([]).recommend(POP_USER) == []


def test_recommend_ties_keep_catalog_order():
    twins = [make_song(1, "First"), make_song(2, "Second"), make_song(3, "Third")]
    results = Recommender(twins).recommend(POP_USER, k=3)
    assert [s.title for s in results] == ["First", "Second", "Third"]


def test_recommend_does_not_modify_the_catalog():
    songs = [make_song(1, "Far", energy=0.1), make_song(2, "Near", energy=0.8)]
    rec = Recommender(songs)
    rec.recommend(POP_USER, k=2)
    assert [s.title for s in rec.songs] == ["Far", "Near"]


def test_acoustic_preference_changes_the_ranking():
    songs = [
        make_song(1, "Plugged", genre="x", mood="y", energy=0.5, acousticness=0.05),
        make_song(2, "Unplugged", genre="x", mood="y", energy=0.5, acousticness=0.95),
    ]
    rec = Recommender(songs)
    likes = UserProfile("pop", "happy", 0.5, likes_acoustic=True)
    dislikes = UserProfile("pop", "happy", 0.5, likes_acoustic=False)
    assert rec.recommend(likes, k=1)[0].title == "Unplugged"
    assert rec.recommend(dislikes, k=1)[0].title == "Plugged"


def test_recommend_cleans_messy_profile_labels():
    """' POP ' and 'Happy' must still earn the genre and mood matches."""
    rec = Recommender([
        make_song(1, "Other", genre="rock", mood="intense", energy=0.8, acousticness=0.0),
        make_song(2, "Match", genre="pop", mood="happy", energy=0.8, acousticness=0.0),
    ])
    messy = UserProfile(" POP ", "Happy", 0.8, likes_acoustic=False)
    assert rec.recommend(messy, k=1)[0].title == "Match"


def test_recommend_clamps_out_of_range_energy():
    rec = Recommender([make_song(1, "Loud", energy=1.0), make_song(2, "Soft", energy=0.2)])
    wild = UserProfile("pop", "happy", 5.0, likes_acoustic=False)
    assert rec.recommend(wild, k=1)[0].title == "Loud"


def test_recommender_matches_functional_api_on_real_catalog():
    """The class and recommend_songs() must agree on the real data."""
    rows = load_songs("data/songs.csv")
    songs = [Song(**row) for row in rows]
    user = UserProfile("lofi", "chill", 0.4, likes_acoustic=True)
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True}

    from_class = [s.title for s in Recommender(songs).recommend(user, k=5)]
    from_function = [song["title"] for song, _, _ in recommend_songs(prefs, rows, k=5)]
    assert from_class == from_function
    assert from_class[0] == "Library Rain"


# ---- Recommender.explain_recommendation ----

def test_explain_lists_matches_and_score_for_a_perfect_fit():
    song = make_song(1, "Perfect", genre="pop", mood="happy", energy=0.8, acousticness=0.0)
    text = Recommender([song]).explain_recommendation(POP_USER, song)
    assert "mood match" in text
    assert "genre match" in text
    assert "energy 0.80 vs target 0.80" in text
    assert text.startswith("1.00 / 1.00")


def test_explain_omits_matches_that_did_not_happen():
    song = make_song(1, "Miss", genre="jazz", mood="sad", energy=0.8, acousticness=0.5)
    text = Recommender([song]).explain_recommendation(POP_USER, song)
    assert "mood match" not in text
    assert "genre match" not in text


def test_explain_score_agrees_with_score_song():
    song = make_song(1, "Mid", genre="pop", mood="sad", energy=0.6, acousticness=0.3)
    expected, _ = score_song(
        {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}, asdict(song)
    )
    text = Recommender([song]).explain_recommendation(POP_USER, song)
    assert text.startswith(f"{expected:.2f} / 1.00")
