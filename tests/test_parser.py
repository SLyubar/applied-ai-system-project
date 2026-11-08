import pytest

from src.parser import parse_query

GENRES = ["ambient", "classical", "edm", "hip hop", "indie pop", "lofi", "pop", "rock"]
MOODS = ["chill", "euphoric", "happy", "intense", "melancholy", "relaxed", "romantic"]


def parse(text):
    return parse_query(text, GENRES, MOODS)


def test_mellow_acoustic_study_request():
    profile, notes = parse("something mellow and acoustic to study to")
    assert profile["genre"] == "lofi"
    assert profile["mood"] == "chill"
    assert profile["energy"] == 0.35
    assert profile["likes_acoustic"] is True
    assert notes == []


def test_synonyms_map_to_catalog_moods():
    assert parse("I feel sad tonight")[0]["mood"] == "melancholy"
    assert parse("cheerful songs")[0]["mood"] == "happy"


def test_literal_catalog_labels_are_recognized():
    profile, _ = parse("high energy rock that feels intense")
    assert profile["genre"] == "rock"
    assert profile["mood"] == "intense"
    assert profile["energy"] == 0.80


def test_longest_genre_wins():
    assert parse("some indie pop please")[0]["genre"] == "indie pop"
    assert parse("some pop please")[0]["genre"] == "pop"


def test_whole_word_matching_avoids_false_positives():
    profile, _ = parse("brunch playlist")  # must not match the 'run' energy cue
    assert profile["energy"] == 0.5


def test_negated_acoustic_means_not_acoustic():
    profile, _ = parse("pop that is not acoustic")
    assert profile["likes_acoustic"] is False


def test_no_acoustic_cue_omits_the_preference():
    profile, _ = parse("upbeat pop")
    assert "likes_acoustic" not in profile


def test_workout_request_gets_high_energy():
    profile, _ = parse("workout music")
    assert profile["energy"] == 0.90


def test_unrecognized_request_returns_empty_labels_and_notes():
    profile, notes = parse("xyzzy")
    assert profile["genre"] == "" and profile["mood"] == ""
    assert profile["energy"] == 0.5
    assert len(notes) == 3


def test_only_catalog_labels_are_ever_returned():
    # 'jazz' is a plausible genre word but is not in this catalog.
    profile, _ = parse("smooth jazz")
    assert profile["genre"] == ""


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_query_raises(empty):
    with pytest.raises(ValueError):
        parse(empty)
