"""Tests for src/pipeline.py: the CLI's query flow returned as data."""

import pytest

from src import pipeline
from src.pipeline import run_query
from src.recommender import load_songs


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def test_keyword_run_returns_profile_ranked_songs_and_confidence(songs):
    result = run_query("calm lofi to study to", songs)
    assert result["parser"] == "keyword" and result["retriever"] == "none"
    assert result["profile"]["genre"] == "lofi"
    recs = result["recommendations"]
    assert len(recs) == 5
    assert [r["score"] for r in recs] == sorted((r["score"] for r in recs), reverse=True)
    assert {"id", "title", "artist", "score", "reasons"} <= set(recs[0])
    assert result["confidence"].label in {"high", "medium", "low"}


def test_k_limits_the_number_of_songs(songs):
    assert len(run_query("calm music", songs, k=3)["recommendations"]) == 3


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_raises(songs, query):
    with pytest.raises(ValueError):
        run_query(query, songs)


def test_custom_parse_fn_is_used(songs):
    seen = []

    def parse(query, genres, moods):
        seen.append(query)
        return {"genre": "rock", "mood": "intense", "energy": 0.9}, ["custom parser"]

    result = run_query("anything", songs, parse_fn=parse)
    assert seen == ["anything"] and "custom parser" in result["notes"]
    assert result["profile"]["genre"] == "rock"


def test_unavailable_parser_falls_back_to_keywords_and_says_so(songs, monkeypatch):
    real = pipeline.get_parse_fn

    def fake(name):
        if name == "gemini":
            raise RuntimeError("no API key")
        return real(name)

    monkeypatch.setattr(pipeline, "get_parse_fn", fake)
    result = run_query("calm music", songs, parser="gemini")
    assert result["parser"] == "keyword"
    assert any("gemini parser unavailable" in n and "no API key" in n for n in result["notes"])


def test_tfidf_retrieval_restricts_the_pool_and_is_noted(songs):
    result = run_query("calm music to study to", songs, retriever="tfidf", pool=8)
    assert result["retriever"] == "tfidf"
    assert any(n.startswith("retrieved 8 candidates (tfidf)") for n in result["notes"])


def test_failed_retrieval_ranks_whole_catalog_and_says_so(songs, monkeypatch):
    def boom(name):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(pipeline, "get_embedder", boom)
    result = run_query("calm music", songs, retriever="gemini")
    assert result["retriever"] == "none"
    assert any("retrieval unavailable" in n and "quota exceeded" in n for n in result["notes"])
    assert len(result["recommendations"]) == 5
