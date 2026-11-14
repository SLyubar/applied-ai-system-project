import pytest

from src.confidence import (
    HIGH,
    MEDIUM,
    analyze_request,
    assess,
    calibration_table,
    label_for,
    understanding,
)
from src.guardrails import known_labels
from src.recommender import load_songs


@pytest.fixture(scope="module")
def labels():
    return known_labels(load_songs("data/songs.csv"))


@pytest.fixture(scope="module")
def genres(labels):
    return labels[0]


@pytest.fixture(scope="module")
def moods(labels):
    return labels[1]


FULL = {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True}


# ---- bands ----

@pytest.mark.parametrize("score,label", [
    (1.0, "high"), (HIGH, "high"), (HIGH - 0.001, "medium"), (MEDIUM, "medium"), (MEDIUM - 0.001, "low"), (0.0, "low"),
])
def test_label_boundaries(score, label):
    assert label_for(score) == label


# ---- what was specified and understood ----

def test_a_complete_profile_specifies_and_understands_everything(genres, moods):
    specified, understood, reasons = analyze_request(FULL, [], genres, moods)
    assert specified == understood == {"energy", "mood", "genre", "acoustic"}
    assert reasons == []


def test_an_unknown_label_is_specified_but_not_understood(genres, moods):
    profile = {**FULL, "genre": "vaporwave"}
    specified, understood, reasons = analyze_request(profile, [], genres, moods)
    assert "genre" in specified and "genre" not in understood
    assert reasons == ["genre 'vaporwave' is not in the catalog"]


def test_an_empty_label_is_unspecified_not_misunderstood(genres, moods):
    profile = {"genre": "", "mood": "", "energy": 0.9}
    specified, understood, reasons = analyze_request(profile, [], genres, moods)
    assert specified == understood == {"energy"}
    assert reasons == []


def test_a_label_the_parser_dropped_still_counts_as_requested(genres, moods):
    profile = {"genre": "", "mood": "chill", "energy": 0.4}
    notes = ["Gemini returned genre 'vaporwave', which is not in the catalog; ignored"]
    specified, understood, reasons = analyze_request(profile, notes, genres, moods)
    assert "genre" in specified and "genre" not in understood
    assert reasons == ["a genre in the request is not in the catalog"]


def test_a_defaulted_energy_is_not_a_stated_criterion(genres, moods):
    notes = ["no energy cue found; defaulted to 0.5"]
    specified, understood, _ = analyze_request({"genre": "lofi", "mood": "", "energy": 0.5}, notes, genres, moods)
    assert "energy" not in specified and specified == {"genre"}


def test_labels_are_matched_after_normalization(genres, moods):
    _, understood, _ = analyze_request({"genre": " Lofi ", "mood": "CHILL", "energy": 0.4}, [], genres, moods)
    assert {"genre", "mood"} <= understood


# ---- understanding ----

def test_full_understanding_is_one(genres, moods):
    assert understanding(FULL, [], genres, moods) == (1.0, [])


def test_understanding_is_weighted_like_the_scorer(genres, moods):
    profile = {"genre": "vaporwave", "mood": "chill", "energy": 0.4, "likes_acoustic": True}
    score, _ = understanding(profile, [], genres, moods)
    assert score == pytest.approx(0.80)  # genre (0.20) of a total 1.00 was not understood


def test_a_request_with_nothing_usable_scores_zero(genres, moods):
    profile = {"genre": "", "mood": "", "energy": 0.5}
    score, reasons = understanding(profile, ["no energy cue found; defaulted to 0.5"], genres, moods)
    assert score == 0.0
    assert "nothing in the request could be turned into a criterion" in reasons


def test_parser_certainty_can_lower_understanding(genres, moods):
    score, reasons = understanding({**FULL, "parser_confidence": 0.4}, [], genres, moods)
    assert score == pytest.approx(0.4)
    assert any("low certainty (0.40)" in r for r in reasons)


def test_parser_certainty_can_never_raise_understanding(genres, moods):
    profile = {**FULL, "genre": "vaporwave", "parser_confidence": 1.0}
    assert understanding(profile, [], genres, moods)[0] == pytest.approx(0.80)


def test_moderate_parser_certainty_lowers_the_score_without_a_warning(genres, moods):
    score, reasons = understanding({**FULL, "parser_confidence": 0.7}, [], genres, moods)
    assert score == pytest.approx(0.7) and reasons == []


@pytest.mark.parametrize("reported,expected", [(5.0, 1.0), (-1.0, 0.0)])
def test_out_of_range_parser_certainty_is_clamped(genres, moods, reported, expected):
    assert understanding({**FULL, "parser_confidence": reported}, [], genres, moods)[0] == expected


def test_a_non_numeric_parser_certainty_is_ignored(genres, moods):
    for junk in ("high", True, None):
        assert understanding({**FULL, "parser_confidence": junk}, [], genres, moods)[0] == 1.0


# ---- overall confidence ----

def test_a_clear_request_with_a_great_match_is_high(genres, moods):
    c = assess(FULL, [], genres, moods, top_score=0.96)
    assert c.label == "high" and c.score == pytest.approx(0.96)
    assert c.reasons == []
    assert c.criteria == ["energy", "mood", "genre", "acoustic"]


def test_unknown_labels_give_low_confidence_instead_of_a_confident_looking_score(genres, moods):
    """The vaporwave/sad profile used to show 0.48 with no warning of any kind."""
    profile = {"genre": "vaporwave", "mood": "sad", "energy": 0.5, "likes_acoustic": False}
    c = assess(profile, [], genres, moods, top_score=0.48)
    assert c.label == "low"
    assert "genre 'vaporwave' is not in the catalog" in c.reasons
    assert "mood 'sad' is not in the catalog" in c.reasons


def test_a_request_that_only_states_some_criteria_can_still_be_high(genres, moods):
    """'workout music, not acoustic' specifies energy and acoustic only; matching both well is high."""
    profile = {"genre": "", "mood": "", "energy": 0.9, "likes_acoustic": False}
    c = assess(profile, [], genres, moods, top_score=0.53)
    assert c.label == "high"
    assert c.criteria == ["energy", "acoustic"]


def test_a_weak_best_match_is_low_even_when_everything_was_understood(genres, moods):
    c = assess(FULL, [], genres, moods, top_score=0.3)
    assert c.understanding == 1.0 and c.label == "low"
    assert any("fits only 30%" in r for r in c.reasons)


def test_a_contradictory_profile_lands_in_the_middle(genres, moods):
    profile = {"genre": "pop", "mood": "chill", "energy": 0.9, "likes_acoustic": False}
    assert assess(profile, [], genres, moods, top_score=0.73).label == "medium"


def test_confidence_is_the_weaker_of_understanding_and_match(genres, moods):
    c = assess({**FULL, "parser_confidence": 0.65}, [], genres, moods, top_score=0.96)
    assert c.score == pytest.approx(min(c.understanding, c.match)) == pytest.approx(0.65)


def test_gibberish_is_low(genres, moods):
    profile = {"genre": "", "mood": "", "energy": 0.5}
    c = assess(profile, ["no energy cue found; defaulted to 0.5"], genres, moods, top_score=0.4)
    assert c.label == "low" and c.score == 0.0


def test_no_songs_means_low_confidence(genres, moods):
    c = assess(FULL, [], genres, moods, top_score=None)
    assert c.label == "low" and "there were no songs to recommend" in c.reasons


def test_score_is_always_between_zero_and_one(genres, moods):
    for top in (-1.0, 0.0, 0.5, 1.0, 5.0):
        assert 0.0 <= assess(FULL, [], genres, moods, top_score=top).score <= 1.0


def test_as_dict_is_json_friendly_and_rounded(genres, moods):
    d = assess(FULL, [], genres, moods, top_score=0.987654).as_dict()
    assert d["label"] == "high" and d["score"] == 0.988
    assert set(d) == {"score", "label", "understanding", "match", "criteria", "reasons"}


# ---- calibration ----

def test_calibration_table_counts_pass_rates_per_band():
    rows = [("high", True), ("high", True), ("high", False), ("medium", True), ("low", False), ("low", False)]
    table = {r["band"]: r for r in calibration_table(rows)}
    assert (table["high"]["cases"], table["high"]["passed"]) == (3, 2)
    assert table["high"]["pass_rate"] == pytest.approx(2 / 3)
    assert table["medium"]["pass_rate"] == 1.0
    assert table["low"]["pass_rate"] == 0.0


def test_calibration_table_has_no_rate_for_an_empty_band():
    table = {r["band"]: r for r in calibration_table([("high", True)])}
    assert table["low"]["cases"] == 0 and table["low"]["pass_rate"] is None
    assert [r["band"] for r in calibration_table([])] == ["high", "medium", "low"]
