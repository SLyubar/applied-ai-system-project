"""Retrieval over song descriptions (the RAG step).

A Retriever embeds every song's description once, then finds the songs whose
descriptions are closest in meaning to a free-text query. The retrieved songs are the
*candidates*; the deterministic scorer then ranks only those, so retrieval widens
what a request can match without changing how songs are scored or explained.

Two interchangeable embedders:
- TfidfEmbedder: local, offline, deterministic, free. Matches on shared words.
- GeminiEmbedder: semantic (understands that "warm hug" is close to "gentle, cozy"),
  but needs an API key and network. Vectors are cached on disk so each text is
  embedded at most once.
"""

import hashlib
import json
import math
import os
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"  # one vector per input text; override with GEMINI_EMBEDDING_MODEL
EMBEDDING_DIMENSIONS = 768  # reduced from the 3072 default; vectors are re-normalized here
CACHE_PATH = "cache/embeddings.json"
EMBED_BATCH_SIZE = 20
MAX_RATE_LIMIT_RETRIES = 3  # the free tier allows about 100 embedding requests per minute
MAX_RETRY_WAIT_SECONDS = 90.0  # a longer suggested wait means a daily quota, which retrying cannot fix
REQUEST_TIMEOUT_MS = 60_000  # never wait on a dead connection forever

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is it its of on or "
    "our she so than that the their them then there these they this to up was we were what when "
    "which while who will with you your".split()
)


def song_text(song: Dict) -> str:
    """The text that represents a song for retrieval: its description plus genre and mood."""
    return f"{song['description']} ({song['genre']}, {song['mood']})"


def tokenize(text: str) -> List[str]:
    """Lowercase words with a light suffix strip, so 'studying' and 'study' share a stem."""
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        for suffix in ("ing", "es", "ed", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
        tokens.append(word)
    return tokens


def normalize(vector: Sequence[float]) -> List[float]:
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else list(vector)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are already unit length


class TfidfEmbedder:
    """Bag-of-words TF-IDF vectors. The vocabulary is fixed by the documents it is fitted on."""

    name = "tfidf"

    def __init__(self) -> None:
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []

    def _vector(self, tokens: Sequence[str]) -> List[float]:
        vector = [0.0] * len(self.vocabulary)
        for token in tokens:
            index = self.vocabulary.get(token)
            if index is not None:
                vector[index] += 1.0
        return normalize([count * self.idf[i] for i, count in enumerate(vector)])

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        tokenized = [tokenize(t) for t in texts]
        document_frequency: Dict[str, int] = {}
        for tokens in tokenized:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.vocabulary = {token: i for i, token in enumerate(sorted(document_frequency))}
        n = len(texts)
        self.idf = [math.log((1 + n) / (1 + document_frequency[t])) + 1.0 for t in self.vocabulary]
        return [self._vector(tokens) for tokens in tokenized]

    def embed_query(self, text: str) -> List[float]:
        if not self.vocabulary:
            raise RuntimeError("embed_documents must be called before embed_query")
        return self._vector(tokenize(text))


def retry_delay_seconds(error: Exception, default: float = 45.0) -> float:
    """How long the API asks us to wait after a rate-limit error, in seconds.

    Reads hints like 'retry in 40.9s' or 'retry in 15h14m22s' and adds a second of margin. Falls back
    to `default` when the message has no hint.
    """
    match = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", str(error), re.IGNORECASE)
    if not match or not any(match.groups()):
        return default
    hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds + 1.0


_DEFAULT_CACHE = object()  # sentinel: use the module-level CACHE_PATH


class GeminiEmbedder:
    """Gemini embeddings (gemini-embedding-001) with an on-disk cache.

    Documents use the RETRIEVAL_DOCUMENT task type and queries use RETRIEVAL_QUERY, as
    the API docs recommend. Credentials come from GEMINI_API_KEY or GOOGLE_API_KEY.
    """

    name = "gemini"

    def __init__(
        self,
        client=None,
        model: Optional[str] = None,
        cache_path: Optional[str] = _DEFAULT_CACHE,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._sleep = sleep
        self.model = model or os.environ.get("GEMINI_EMBEDDING_MODEL") or GEMINI_EMBEDDING_MODEL
        # Resolved at construction (not definition) so the module default can be changed or patched.
        self.cache_path = CACHE_PATH if cache_path is _DEFAULT_CACHE else cache_path
        self._cache: Optional[Dict[str, List[float]]] = None
        self.api_calls = 0

    # -- cache --
    def _key(self, task: str, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{task}|{EMBEDDING_DIMENSIONS}|{text}".encode("utf-8")).hexdigest()

    def _load_cache(self) -> Dict[str, List[float]]:
        if self._cache is None:
            self._cache = {}
            if self.cache_path and os.path.exists(self.cache_path):
                try:
                    with open(self.cache_path, encoding="utf-8") as f:
                        self._cache = json.load(f)
                except (OSError, json.JSONDecodeError):
                    self._cache = {}  # a damaged cache is rebuilt, never fatal
        return self._cache

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        folder = os.path.dirname(self.cache_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    # -- API --
    def _get_client(self):
        if self._client is None:
            from google import genai  # imported lazily so the TF-IDF path never needs the SDK
            from google.genai import types

            # Raises ValueError if no API key is configured.
            self._client = genai.Client(http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
        return self._client

    def _embed(self, texts: Sequence[str], task: str) -> List[List[float]]:
        cache = self._load_cache()
        keys = [self._key(task, t) for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cache]
        if missing:
            from google.genai import types

            from google.genai import errors as genai_errors

            client = self._get_client()
            for start in range(0, len(missing), EMBED_BATCH_SIZE):
                batch = missing[start:start + EMBED_BATCH_SIZE]
                for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                    try:
                        response = client.models.embed_content(
                            model=self.model,
                            contents=[t for _, t in batch],
                            config=types.EmbedContentConfig(
                                task_type=task, output_dimensionality=EMBEDDING_DIMENSIONS
                            ),
                        )
                        break
                    except genai_errors.ClientError as error:
                        wait = retry_delay_seconds(error) if getattr(error, "code", None) == 429 else None
                        # Give up on anything but a short per-minute limit: a daily quota will not clear by waiting.
                        if wait is None or wait > MAX_RETRY_WAIT_SECONDS or attempt == MAX_RATE_LIMIT_RETRIES:
                            self._save_cache()  # keep whatever was already embedded
                            raise
                        self._sleep(wait)
                self.api_calls += 1
                vectors = [list(e.values) for e in response.embeddings]
                if len(vectors) != len(batch):
                    raise RuntimeError(f"asked for {len(batch)} embeddings, got {len(vectors)}")
                for (key, _), vector in zip(batch, vectors):
                    cache[key] = normalize(vector)  # reduced-size vectors are not normalized by the API
            self._save_cache()
        return [cache[k] for k in keys]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class Retriever:
    """Finds the songs whose descriptions are closest to a query."""

    def __init__(self, songs: List[Dict], embedder, text_fn: Callable[[Dict], str] = song_text) -> None:
        if not songs:
            raise ValueError("cannot build a retriever over an empty catalog")
        self.songs = songs
        self.embedder = embedder
        self.ids = [s["id"] for s in songs]
        self.vectors = embedder.embed_documents([text_fn(s) for s in songs])

    def search(self, query: str, n: int) -> List[Tuple[int, float]]:
        """Return up to n (song id, similarity) pairs, most similar first. Ties keep catalog order."""
        if n <= 0:
            return []
        query_vector = self.embedder.embed_query(query)
        scored = [(song_id, cosine(query_vector, vector)) for song_id, vector in zip(self.ids, self.vectors)]
        scored.sort(key=lambda pair: pair[1], reverse=True)  # stable sort
        return scored[:n]


def make_retrieval_rank_fn(retriever: Retriever, candidates: int, k_floor: int = 5):
    """Build a rank_fn that ranks only the retrieved candidates.

    The returned function takes the query text as an extra `query` argument. With no
    query (a ready-made profile) it ranks the whole catalog, exactly like the baseline.
    The candidate count never drops below k.
    """
    from src.guardrails import known_labels, sanitize_profile
    from src.recommender import recommend_songs

    def rank(profile: Dict, songs: List[Dict], k: int = k_floor, query: Optional[str] = None):
        genres, moods = known_labels(songs)
        clean, warnings = sanitize_profile(profile, genres, moods)
        pool = songs
        if query and candidates < len(songs):
            keep = {song_id for song_id, _ in retriever.search(query, max(candidates, k))}
            pool = [s for s in songs if s["id"] in keep]
        return [song["id"] for song, _, _ in recommend_songs(clean, pool, k=k)], warnings

    rank.wants_query = True
    return rank


def get_embedder(name: str):
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "gemini":
        return GeminiEmbedder()
    raise ValueError(f"unknown embedder {name!r}; expected 'tfidf' or 'gemini'")
