import pytest

from src.ablation import build_variants, evaluate_variants, format_report, main, weighted_rank_fn
from src.evaluation import baseline_rank, evaluate_ranking, load_golden, summarize
from src.recommender import WEIGHTS, load_songs, recommend_songs, score_song


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture(scope="module")
def cases():
    return load_golden()["ranking_cases"]


# ---- the scorer accepts weights without changing its default behavior ----

def test_score_song_default_weights_are_unchanged(songs):
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True}
    song = next(s for s in songs if s["title"] == "Library Rain")
    assert score_song(prefs, song) == score_song(prefs, song, WEIGHTS)


def test_custom_weights_change_the_score(songs):
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True}
    song = next(s for s in songs if s["title"] == "Library Rain")
    only_genre = {"energy": 0.0, "mood": 0.0, "genre": 1.0, "acoustic": 0.0}
    score, reasons = score_song(prefs, song, only_genre)
    assert score == pytest.approx(1.0)
    assert any("genre match (+1.00)" in r for r in reasons)


def test_recommend_songs_passes_weights_through(songs):
    prefs = {"genre": "metal", "mood": "chill", "energy": 0.1, "likes_acoustic": True}
    default_top = recommend_songs(prefs, songs, k=1)[0][0]["genre"]
    genre_only = {"energy": 0.0, "mood": 0.0, "genre": 1.0, "acoustic": 0.0}
    assert recommend_songs(prefs, songs, k=1, weights=genre_only)[0][0]["genre"] == "metal"
    assert default_top != "metal"


# ---- variants ----

def test_variants_cover_ablations_equal_weights_and_perturbations():
    variants = dict(build_variants(WEIGHTS))
    assert len(variants) == 1 + 4 + 1 + 8
    assert variants["Baseline"] == WEIGHTS
    for feature in WEIGHTS:
        assert variants[f"Without {feature}"][feature] == 0.0
        assert variants[f"{feature.capitalize()} x1.5"][feature] == pytest.approx(WEIGHTS[feature] * 1.5)
        assert variants[f"{feature.capitalize()} x0.5"][feature] == pytest.approx(WEIGHTS[feature] * 0.5)
    assert len(set(variants["Equal weights"].values())) == 1


def test_variants_do_not_modify_the_base_weights():
    before = dict(WEIGHTS)
    build_variants(WEIGHTS)
    assert WEIGHTS == before


# ---- evaluation ----

def test_baseline_variant_matches_the_normal_evaluation(cases, songs):
    results = evaluate_variants(cases, songs)
    profile_cases = [c for c in cases if "profile" in c]
    normal = summarize(evaluate_ranking(profile_cases, songs, baseline_rank), [])
    base = results[0]
    assert base["name"] == "Baseline"
    assert base["ndcg"] == pytest.approx(normal["mean_ndcg"])
    assert base["passed"] == normal["ranking_cases_passed"]
    assert base["top1_changed"] == 0


def test_only_profile_cases_are_used(cases, songs):
    results = evaluate_variants(cases, songs)
    assert results[0]["total"] == sum(1 for c in cases if "profile" in c)


def test_dropping_genre_hurts_ranking_quality(cases, songs):
    results = {r["name"]: r for r in evaluate_variants(cases, songs)}
    assert results["Without genre"]["ndcg"] < results["Baseline"]["ndcg"]


def test_weighted_rank_fn_with_default_weights_equals_baseline(cases, songs):
    rank = weighted_rank_fn(WEIGHTS)
    for case in (c for c in cases if "profile" in c):
        assert rank(case["profile"], songs, 5) == baseline_rank(case["profile"], songs, 5)


# ---- report ----

def test_report_lists_every_variant_and_a_caveat(cases, songs):
    report = format_report(evaluate_variants(cases, songs), songs)
    for name in ("Baseline", "Without energy", "Without mood", "Equal weights", "Genre x1.5"):
        assert f"| {name} |" in report
    assert "written by one person" in report


def test_main_writes_the_report(tmp_path, capsys):
    out = tmp_path / "ablation.md"
    assert main(["--out", str(out)]) == 0
    assert "# Weight sensitivity and ablation" in out.read_text(encoding="utf-8")
    assert "Baseline" in capsys.readouterr().out
