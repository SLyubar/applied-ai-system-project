"""Tests for retrieval. No network or API key is needed: Gemini is replaced by a fake client."""

import json
import math

import pytest
from google.genai import errors as genai_errors

from src import retrieval
from src.evaluation import baseline_rank, evaluate_ranking, load_golden
from src.main import main as cli_main, print_recommendations
from src.recommender import load_songs
from src.retrieval import (
    EMBED_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    GEMINI_EMBEDDING_MODEL,
    MAX_RETRY_WAIT_SECONDS,
    GeminiEmbedder,
    Retriever,
    TfidfEmbedder,
    cosine,
    get_embedder,
    make_retrieval_rank_fn,
    normalize,
    retry_delay_seconds,
    song_text,
    tokenize,
)


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def tiny_catalog():
    def song(i, genre, mood, description):
        return {"id": i, "title": f"Song {i}", "artist": "A", "genre": genre, "mood": mood, "energy": 0.5,
                "tempo_bpm": 100, "valence": 0.5, "danceability": 0.5, "acousticness": 0.5,
                "description": description}

    return [
        song(1, "lofi", "chill", "gentle rainy piano for studying"),
        song(2, "metal", "angry", "furious thrash guitars and blast beats"),
        song(3, "jazz", "relaxed", "lazy brushed drums for a cafe afternoon"),
        song(4, "edm", "euphoric", "massive festival drops and synth builds"),
    ]


# ---- text helpers ----

def test_tokenize_lowercases_drops_stopwords_and_strips_suffixes():
    assert tokenize("The Studying of Rainy days") == ["study", "rainy", "day"]


def test_tokenize_leaves_short_words_alone():
    assert tokenize("bus gas") == ["bus", "gas"]


def test_tokenize_handles_empty_and_non_latin_text():
    assert tokenize("") == []
    assert tokenize("雨の日") == []  # no Latin tokens: the TF-IDF embedder cannot read it, by design


def test_song_text_combines_description_genre_and_mood():
    assert song_text({"description": "soft piano", "genre": "classical", "mood": "calm"}) == "soft piano (classical, calm)"


def test_normalize_gives_unit_length_and_tolerates_zero():
    assert math.sqrt(sum(x * x for x in normalize([3.0, 4.0]))) == pytest.approx(1.0)
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_cosine_of_a_unit_vector_with_itself_is_one():
    v = normalize([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


# ---- TF-IDF ----

def test_tfidf_documents_are_unit_vectors_and_deterministic():
    texts = ["calm piano music", "loud metal guitars", "calm quiet evening"]
    first = TfidfEmbedder().embed_documents(texts)
    second = TfidfEmbedder().embed_documents(texts)
    assert first == second
    for v in first:
        assert sum(x * x for x in v) == pytest.approx(1.0)


def test_tfidf_query_matches_the_document_that_shares_its_words():
    embedder = TfidfEmbedder()
    docs = embedder.embed_documents(["calm piano music", "loud metal guitars"])
    query = embedder.embed_query("piano")
    assert cosine(query, docs[0]) > cosine(query, docs[1])


def test_tfidf_gives_rare_words_more_weight_than_common_ones():
    embedder = TfidfEmbedder()
    embedder.embed_documents(["music piano", "music guitar", "music drums"])
    assert embedder.idf[embedder.vocabulary["piano"]] > embedder.idf[embedder.vocabulary["music"]]


def test_tfidf_ignores_words_it_has_never_seen():
    embedder = TfidfEmbedder()
    embedder.embed_documents(["calm piano"])
    assert embedder.embed_query("xylophone zeppelin") == [0.0, 0.0]


def test_tfidf_query_before_fitting_is_an_error():
    with pytest.raises(RuntimeError):
        TfidfEmbedder().embed_query("anything")


# ---- Retriever ----

def test_retriever_ranks_the_matching_song_first():
    retriever = Retriever(tiny_catalog(), TfidfEmbedder())
    [(top_id, similarity), *_] = retriever.search("rainy piano for studying", 4)
    assert top_id == 1 and similarity > 0.3


def test_retriever_returns_at_most_n_results_best_first():
    retriever = Retriever(tiny_catalog(), TfidfEmbedder())
    results = retriever.search("festival drops", 2)
    assert len(results) == 2
    assert results[0][0] == 4
    assert results[0][1] >= results[1][1]


def test_retriever_with_non_positive_n_returns_nothing():
    retriever = Retriever(tiny_catalog(), TfidfEmbedder())
    assert retriever.search("anything", 0) == []
    assert retriever.search("anything", -3) == []


def test_ties_keep_catalog_order():
    retriever = Retriever(tiny_catalog(), TfidfEmbedder())
    results = retriever.search("xylophone", 4)  # matches nothing, so every similarity is 0
    assert [song_id for song_id, _ in results] == [1, 2, 3, 4]


def test_empty_catalog_is_rejected():
    with pytest.raises(ValueError):
        Retriever([], TfidfEmbedder())


# ---- the retrieve-then-score pipeline ----

def test_rank_fn_only_returns_retrieved_candidates(songs):
    retriever = Retriever(songs, TfidfEmbedder())
    rank = make_retrieval_rank_fn(retriever, 10)
    query = "gentle rainy piano for studying"
    candidates = {song_id for song_id, _ in retriever.search(query, 10)}
    profile = {"genre": "metal", "mood": "angry", "energy": 0.95}  # the scorer alone would pick metal
    ranked, _ = rank(profile, songs, 5, query=query)
    assert set(ranked) <= candidates


def test_rank_fn_without_a_query_matches_the_baseline(songs):
    rank = make_retrieval_rank_fn(Retriever(songs, TfidfEmbedder()), 10)
    profile = {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True}
    assert rank(profile, songs, 5) == baseline_rank(profile, songs, 5)


def test_pool_larger_than_the_catalog_changes_nothing(songs):
    rank = make_retrieval_rank_fn(Retriever(songs, TfidfEmbedder()), 500)
    profile = {"genre": "lofi", "mood": "chill", "energy": 0.4}
    assert rank(profile, songs, 5, query="rainy piano")[0] == baseline_rank(profile, songs, 5)[0]


def test_pool_never_shrinks_below_k(songs):
    rank = make_retrieval_rank_fn(Retriever(songs, TfidfEmbedder()), 1)
    ranked, _ = rank({"genre": "lofi", "mood": "chill", "energy": 0.4}, songs, 5, query="rainy piano")
    assert len(ranked) == 5


def test_rank_fn_still_cleans_the_profile(songs):
    rank = make_retrieval_rank_fn(Retriever(songs, TfidfEmbedder()), 10)
    _, warnings = rank({"genre": "Lofi ", "mood": "chill", "energy": 1.5}, songs, 5, query="piano")
    assert any("clamped" in w for w in warnings)


def test_harness_passes_the_query_only_to_retrieval_aware_functions(songs):
    seen = []

    def aware(profile, songs, k, query=None):
        seen.append(query)
        return [1], []

    aware.wants_query = True

    def plain(profile, songs, k):
        return [1], []

    cases = [{"name": "q", "query": "calm music", "acceptable_ids": [1]},
             {"name": "p", "profile": {"genre": "lofi", "mood": "chill", "energy": 0.4}, "acceptable_ids": [1]}]
    evaluate_ranking(cases, songs, aware)
    evaluate_ranking(cases, songs, plain)  # must not be given a query argument
    assert seen == ["calm music", None]


# ---- Gemini embedder (fake client) ----

class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeModels:
    def __init__(self, errors=None, wrong_count=False):
        self.calls, self.errors, self.wrong_count = [], list(errors or []), wrong_count

    def embed_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        if self.errors:
            raise self.errors.pop(0)
        count = len(contents) - 1 if self.wrong_count else len(contents)
        # Deliberately not unit length, like the API's reduced-dimension vectors.
        return FakeResponse([FakeEmbedding([3.0 + i, 4.0, 0.0]) for i in range(count)])


class FakeClient:
    def __init__(self, **kwargs):
        self.models = FakeModels(**kwargs)


def rate_limit(message="Please retry in 2.0s."):
    return genai_errors.ClientError(429, {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED"}})


def embedder(tmp_path, client=None, **kwargs):
    return GeminiEmbedder(client=client or FakeClient(), cache_path=str(tmp_path / "cache" / "e.json"),
                          sleep=lambda s: None, **kwargs)


def test_documents_are_embedded_in_batches(tmp_path):
    e = embedder(tmp_path)
    texts = [f"text {i}" for i in range(EMBED_BATCH_SIZE * 2 + 5)]
    vectors = e.embed_documents(texts)
    assert len(vectors) == len(texts)
    assert [len(c["contents"]) for c in e._client.models.calls] == [EMBED_BATCH_SIZE, EMBED_BATCH_SIZE, 5]


def test_vectors_are_normalized_even_if_the_api_does_not(tmp_path):
    [v] = embedder(tmp_path).embed_documents(["x"])
    assert sum(x * x for x in v) == pytest.approx(1.0)


def test_documents_and_queries_use_different_task_types(tmp_path):
    e = embedder(tmp_path)
    e.embed_documents(["a song"])
    e.embed_query("a request")
    doc_call, query_call = e._client.models.calls
    assert doc_call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert query_call["config"].task_type == "RETRIEVAL_QUERY"
    assert doc_call["config"].output_dimensionality == EMBEDDING_DIMENSIONS
    assert doc_call["model"] == GEMINI_EMBEDDING_MODEL


def test_model_can_be_overridden_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "custom-embedder")
    e = embedder(tmp_path)
    e.embed_query("x")
    assert e._client.models.calls[0]["model"] == "custom-embedder"


def test_repeated_texts_hit_the_cache_and_make_no_api_calls(tmp_path):
    e = embedder(tmp_path)
    first = e.embed_documents(["a", "b"])
    calls_after_first = len(e._client.models.calls)
    assert e.embed_documents(["a", "b"]) == first
    assert len(e._client.models.calls) == calls_after_first


def test_duplicate_texts_in_one_request_are_sent_once(tmp_path):
    e = embedder(tmp_path)
    e.embed_documents(["same", "same", "same"])
    assert e._client.models.calls[0]["contents"] == ["same"]


def test_cache_survives_a_restart(tmp_path):
    first = embedder(tmp_path)
    vectors = first.embed_documents(["a", "b"])
    fresh_client = FakeClient()
    second = GeminiEmbedder(client=fresh_client, cache_path=first.cache_path, sleep=lambda s: None)
    assert second.embed_documents(["a", "b"]) == vectors
    assert fresh_client.models.calls == []


def test_a_damaged_cache_file_is_rebuilt_not_fatal(tmp_path):
    e = embedder(tmp_path)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "e.json").write_text("{broken", encoding="utf-8")
    assert len(e.embed_documents(["a"])) == 1


def test_different_models_do_not_share_cache_entries(tmp_path):
    a = embedder(tmp_path, model="model-a")
    a.embed_documents(["x"])
    client_b = FakeClient()
    b = GeminiEmbedder(client=client_b, model="model-b", cache_path=a.cache_path, sleep=lambda s: None)
    b.embed_documents(["x"])
    assert len(client_b.models.calls) == 1


def test_wrong_number_of_embeddings_is_an_error(tmp_path):
    with pytest.raises(RuntimeError):
        embedder(tmp_path, client=FakeClient(wrong_count=True)).embed_documents(["a", "b"])


def test_rate_limit_is_waited_out_and_retried(tmp_path):
    waits = []
    client = FakeClient(errors=[rate_limit("Please retry in 2.0s.")])
    e = GeminiEmbedder(client=client, cache_path=str(tmp_path / "e.json"), sleep=waits.append)
    assert len(e.embed_documents(["a"])) == 1
    assert waits == [3.0]  # the API's hint plus one second
    assert len(client.models.calls) == 2


def test_persistent_rate_limit_eventually_raises_and_keeps_earlier_work(tmp_path):
    texts = [f"t{i}" for i in range(EMBED_BATCH_SIZE + 1)]
    client = FakeClient()
    e = GeminiEmbedder(client=client, cache_path=str(tmp_path / "e.json"), sleep=lambda s: None)
    # First batch succeeds; every attempt at the second batch is rate limited.
    original = client.models.embed_content

    def flaky(model, contents, config):
        if len(client.models.calls) >= 1:
            client.models.calls.append({})
            raise rate_limit()
        return original(model, contents, config)

    client.models.embed_content = flaky
    with pytest.raises(genai_errors.ClientError):
        e.embed_documents(texts)
    saved = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert len(saved) == EMBED_BATCH_SIZE  # the first batch was kept


def test_other_client_errors_are_not_retried(tmp_path):
    bad = genai_errors.ClientError(400, {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}})
    client = FakeClient(errors=[bad])
    with pytest.raises(genai_errors.ClientError):
        embedder(tmp_path, client=client).embed_documents(["a"])
    assert len(client.models.calls) == 1


def test_missing_credentials_raise_a_clear_error(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError):
        GeminiEmbedder(cache_path=str(tmp_path / "e.json")).embed_query("x")


def test_cached_queries_need_no_credentials(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    warm = embedder(tmp_path)
    vector = warm.embed_query("rainy day")
    cold = GeminiEmbedder(cache_path=warm.cache_path)  # no client, no key
    assert cold.embed_query("rainy day") == vector


# ---- helpers ----

def test_retry_delay_reads_the_apis_hint_and_adds_a_second():
    assert retry_delay_seconds(Exception("Please retry in 40.5s.")) == pytest.approx(41.5)
    assert retry_delay_seconds(Exception("retry in 2m5s")) == pytest.approx(126.0)
    assert retry_delay_seconds(Exception("Please retry in 15h14m22s or upgrade")) == pytest.approx(54863.0)


def test_retry_delay_falls_back_to_a_default_when_there_is_no_hint():
    assert retry_delay_seconds(Exception("rate limited")) == 45.0
    assert retry_delay_seconds(Exception("rate limited"), default=7.0) == 7.0


def test_get_embedder_selects_by_name():
    assert isinstance(get_embedder("tfidf"), TfidfEmbedder)
    assert isinstance(get_embedder("gemini"), GeminiEmbedder)
    with pytest.raises(ValueError):
        get_embedder("word2vec")


# ---- command line ----

def test_query_run_with_a_retriever_shows_candidates_and_records_it(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    cli_main(["--query", "gentle rainy piano", "--retriever", "tfidf", "--pool", "12", "--log-file", str(log)])
    out = capsys.readouterr().out
    assert "retrieved 12 candidates (tfidf)" in out
    assert json.loads(log.read_text(encoding="utf-8"))["retriever"] == "tfidf"


def test_query_run_without_a_retriever_records_none(tmp_path):
    log = tmp_path / "log.jsonl"
    cli_main(["--query", "calm music", "--log-file", str(log)])
    assert json.loads(log.read_text(encoding="utf-8"))["retriever"] == "none"


def test_retrieval_failure_is_reported_and_the_whole_catalog_is_ranked(monkeypatch, capsys):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(retrieval, "CACHE_PATH", None)  # no cache, so nothing can be served without a key
    cli_main(["--query", "calm music", "--retriever", "gemini", "--no-log"])
    out = capsys.readouterr().out
    assert "retrieval unavailable" in out and "ranked the whole catalog" in out
    assert "1. " in out  # results were still produced


def test_print_recommendations_can_be_limited_to_a_pool(songs, capsys):
    pool = [s for s in songs if s["genre"] == "folk"]
    _, recs, _ = print_recommendations("pooled", {"genre": "metal", "mood": "angry", "energy": 0.9}, songs, k=3, pool=pool)
    assert {r["id"] for r in recs} <= {s["id"] for s in pool}


def test_evaluation_cli_accepts_a_retriever(tmp_path):
    from src.evaluation import main as eval_main

    out = tmp_path / "r.md"
    eval_main(["--retriever", "tfidf", "--pool", "15", "--out", str(out)])
    assert "# Evaluation report (baseline + tfidf retrieval (pool 15))" in out.read_text(encoding="utf-8")


def test_evaluation_cli_refuses_an_unusable_retriever(monkeypatch):
    from src.evaluation import main as eval_main

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(retrieval, "CACHE_PATH", None)
    with pytest.raises(SystemExit):
        eval_main(["--retriever", "gemini"])


def test_default_cache_path_is_read_when_the_embedder_is_built(monkeypatch):
    monkeypatch.setattr(retrieval, "CACHE_PATH", "somewhere/else.json")
    assert GeminiEmbedder().cache_path == "somewhere/else.json"
    assert GeminiEmbedder(cache_path=None).cache_path is None


def test_the_embedder_client_is_built_with_a_request_timeout(tmp_path, monkeypatch):
    from google import genai

    built = {}

    def capture(**kwargs):
        built.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(genai, "Client", capture)
    GeminiEmbedder(cache_path=str(tmp_path / "e.json")).embed_query("calm music")
    assert built["http_options"].timeout == retrieval.REQUEST_TIMEOUT_MS


def test_a_daily_quota_is_not_waited_out(tmp_path):
    """'retry in 15h' means the free-tier day is used up. Waiting 90 seconds three times cannot fix that."""
    waits = []
    daily = rate_limit("Rate limit exceeded (limit: 500 requests per day). Please retry in 15h14m22s.")
    client = FakeClient(errors=[daily])
    e = GeminiEmbedder(client=client, cache_path=str(tmp_path / "e.json"), sleep=waits.append)
    with pytest.raises(genai_errors.ClientError):
        e.embed_documents(["a"])
    assert waits == [] and len(client.models.calls) == 1


def test_a_short_per_minute_limit_is_still_retried(tmp_path):
    waits = []
    client = FakeClient(errors=[rate_limit("Please retry in 30.0s.")])
    e = GeminiEmbedder(client=client, cache_path=str(tmp_path / "e.json"), sleep=waits.append)
    assert len(e.embed_documents(["a"])) == 1
    assert waits == [31.0]
