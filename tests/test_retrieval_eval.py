import pytest

from src.evaluation import load_golden
from src.parser import parse_query
from src.recommender import load_songs
from src.retrieval import Retriever, TfidfEmbedder
from src.retrieval_eval import (
    candidate_metrics,
    format_report,
    main,
    memoize_parser,
    retrieval_only_rank_fn,
    run_experiment,
)


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture(scope="module")
def cases():
    return load_golden()["ranking_cases"]


def tfidf_factory(name):
    if name != "tfidf":
        raise ValueError(f"no embedder {name}")
    return TfidfEmbedder()


# ---- memoized parser ----

def test_memoize_parser_calls_the_parser_once_per_query():
    calls = []

    def parser(query, genres, moods):
        calls.append(query)
        return {"genre": "lofi", "energy": 0.3}, ["note"]

    parse = memoize_parser(parser)
    parse("a", [], [])
    parse("a", [], [])
    parse("b", [], [])
    assert calls == ["a", "b"]


def test_memoized_answers_cannot_be_corrupted_by_callers():
    parse = memoize_parser(lambda q, g, m: ({"genre": "lofi"}, ["n"]))
    first, notes = parse("a", [], [])
    first["genre"] = "tampered"
    notes.append("tampered")
    again, again_notes = parse("a", [], [])
    assert again == {"genre": "lofi"} and again_notes == ["n"]


# ---- candidate metrics ----

def test_candidate_metrics_measure_recall_of_relevant_songs(songs):
    retriever = Retriever(songs, TfidfEmbedder())
    case = {"name": "q", "query": "gentle rainy piano for studying", "ideal_ids": [4], "acceptable_ids": [2, 21],
            "forbidden_ids": [17]}
    everything = candidate_metrics(retriever, [case], len(songs))
    assert everything["relevant_recall"] == 1.0 and everything["ideal_recall"] == 1.0
    assert everything["forbidden_candidates"] == 1  # a pool of the whole catalog lets the forbidden song in
    nothing = candidate_metrics(retriever, [case], 0)
    assert nothing["relevant_recall"] == 0.0 and nothing["forbidden_candidates"] == 0


def test_candidate_metrics_skip_cases_with_no_ground_truth(songs):
    retriever = Retriever(songs, TfidfEmbedder())
    assert candidate_metrics(retriever, [{"name": "x", "query": "q"}], 10)["relevant_recall"] == 0.0


def test_smaller_pools_never_recall_more_than_larger_ones(songs, cases):
    retriever = Retriever(songs, TfidfEmbedder())
    queries = [c for c in cases if "query" in c]
    recalls = [candidate_metrics(retriever, queries, n)["relevant_recall"] for n in (5, 10, 20, 40, 80)]
    assert recalls == sorted(recalls) and recalls[-1] == 1.0


# ---- retrieval-only ranking ----

def test_retrieval_only_ranks_by_similarity_without_the_scorer(songs):
    retriever = Retriever(songs, TfidfEmbedder())
    rank = retrieval_only_rank_fn(retriever)
    query = "gentle rainy piano for studying"
    ranked, _ = rank({"genre": "metal", "mood": "angry", "energy": 0.95}, songs, 3, query=query)
    assert ranked == [song_id for song_id, _ in retriever.search(query, 3)]


# ---- the experiment ----

def test_experiment_runs_every_configuration(songs, cases):
    rows, notes = run_experiment(songs, cases, parse_query, ["tfidf"], embedder_factory=tfidf_factory)
    names = [r["name"] for r in rows]
    assert names[0] == "No retrieval (baseline)"
    assert "tfidf: retrieval only (no scorer)" in names
    assert [n for n in names if "then score" in n] == [f"tfidf: retrieve {n}, then score" for n in (10, 20, 40)]
    assert notes == []
    assert rows[0]["candidates"] is None and rows[1]["candidates"] is not None


def test_experiment_only_uses_free_text_cases(songs, cases):
    rows, _ = run_experiment(songs, cases, parse_query, ["tfidf"], embedder_factory=tfidf_factory)
    assert rows[0]["total"] == sum(1 for c in cases if "query" in c)


def test_embedder_that_cannot_be_built_is_skipped_with_a_note(songs, cases):
    rows, notes = run_experiment(songs, cases, parse_query, ["tfidf", "gemini"], embedder_factory=tfidf_factory)
    assert not any(r["name"].startswith("gemini") for r in rows)
    assert any("gemini embedder skipped" in n for n in notes)
    assert any(r["name"].startswith("tfidf") for r in rows)


def test_embedder_that_fails_midway_is_dropped_entirely(songs, cases):
    class Dies(TfidfEmbedder):
        def embed_query(self, text):
            raise RuntimeError("quota exceeded")

    rows, notes = run_experiment(songs, cases, parse_query, ["dies"], embedder_factory=lambda n: Dies())
    assert [r["name"] for r in rows] == ["No retrieval (baseline)"]  # no half-finished rows
    assert any("quota exceeded" in n for n in notes)


def test_the_parser_runs_once_per_query_across_all_configurations(songs, cases):
    calls = []

    def counting(query, genres, moods):
        calls.append(query)
        return parse_query(query, genres, moods)

    run_experiment(songs, cases, counting, ["tfidf"], embedder_factory=tfidf_factory)
    assert len(calls) == len(set(calls)) == sum(1 for c in cases if "query" in c)


def test_baseline_row_matches_a_direct_evaluation(songs, cases):
    from src.evaluation import baseline_rank, evaluate_ranking, summarize

    rows, _ = run_experiment(songs, cases, parse_query, [], embedder_factory=tfidf_factory)
    direct = summarize(evaluate_ranking([c for c in cases if "query" in c], songs, baseline_rank), [])
    assert rows[0]["ndcg"] == pytest.approx(direct["mean_ndcg"])
    assert rows[0]["passed"] == direct["ranking_cases_passed"]


# ---- report ----

def test_report_lists_configurations_and_a_caveat(songs, cases):
    rows, notes = run_experiment(songs, cases, parse_query, ["tfidf", "gemini"], embedder_factory=tfidf_factory)
    report = format_report(rows, notes, songs, "keyword")
    assert "# Retrieval experiment (parser: keyword)" in report
    assert "| No retrieval (baseline) |" in report
    assert "| tfidf: retrieve 20, then score |" in report
    assert "one person's" in report
    assert "gemini embedder skipped" in report


def test_report_warns_when_the_parser_fell_back(songs, cases):
    rows, notes = run_experiment(songs, cases, parse_query, [], embedder_factory=tfidf_factory)
    rows[0]["fallbacks"] = 3
    assert "3 parser answer(s) fell back" in format_report(rows, notes, songs, "gemini")


def test_main_writes_the_report(tmp_path, capsys):
    out = tmp_path / "retrieval.md"
    assert main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "# Retrieval experiment (parser: keyword)" in text
    assert "tfidf: retrieve 10, then score" in text
    assert "Retrieval experiment" in capsys.readouterr().out
