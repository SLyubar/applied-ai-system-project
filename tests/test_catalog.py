"""Data-quality checks for data/songs.csv so a bad row can't silently skew results."""

from collections import Counter

import pytest

from src.recommender import Song, load_songs

NUMERIC_0_TO_1 = ("energy", "valence", "danceability", "acousticness")


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def test_catalog_is_large_enough_for_meaningful_evaluation(songs):
    assert len(songs) >= 60


def test_ids_are_unique(songs):
    ids = [s["id"] for s in songs]
    assert len(ids) == len(set(ids))


def test_no_blank_fields(songs):
    for s in songs:
        for key, value in s.items():
            assert str(value).strip() != "", f"song {s['id']} has blank {key!r}"


def test_labels_are_already_normalized(songs):
    for s in songs:
        assert s["genre"] == s["genre"].strip().lower()
        assert s["mood"] == s["mood"].strip().lower()


@pytest.mark.parametrize("field", NUMERIC_0_TO_1)
def test_features_are_in_range(songs, field):
    for s in songs:
        assert 0.0 <= s[field] <= 1.0, f"song {s['id']} {field}={s[field]}"


def test_tempo_is_plausible(songs):
    for s in songs:
        assert 40 <= s["tempo_bpm"] <= 220, f"song {s['id']} tempo={s['tempo_bpm']}"


def test_every_genre_has_at_least_three_songs(songs):
    counts = Counter(s["genre"] for s in songs)
    thin = {genre: n for genre, n in counts.items() if n < 3}
    assert thin == {}


def test_descriptions_are_meaningful(songs):
    for s in songs:
        assert len(s["description"].split()) >= 6, f"song {s['id']} description too short"


def test_every_row_builds_a_song(songs):
    built = [Song(**row) for row in songs]
    assert len(built) == len(songs)
    assert all(song.description for song in built)


def test_catalog_includes_non_western_genres(songs):
    genres = {s["genre"] for s in songs}
    assert {"afrobeats", "k-pop", "bossa nova"} <= genres
