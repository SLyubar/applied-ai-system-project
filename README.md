# 🎵 Music Recommender, Extended with RAG and a Reliability System

> **DRAFT.** This README describes the planned system. Sections marked **(planned)** describe features that are designed but not yet implemented. Update or remove those markers as each feature lands.

## Original Project: Music Recommender Simulation

This project extends **Music Recommender Simulation**, a small CLI-first content-based music recommender. The original goal was to represent songs and a listener's taste profile as data, turn them into a transparent scoring rule, and evaluate where that rule succeeds and fails. It scored each song in a 20-song CSV on energy closeness, mood match, genre match, and acoustic preference (`0.40 / 0.25 / 0.20 / 0.15`), then returns the top-k songs with a plain-language reason for every point awarded. It was stress-tested against three standard and seven adversarial hand-written profiles.

## Title and Summary

**Music Recommender: Retrieval-Augmented Edition**

This project lets a listener ask for music in plain language ("something calm and acoustic for a rainy study session") and get ranked, explained recommendations. Two additions make that possible: a **Gemini-powered query parser** that turns free text into a structured taste profile, and a **local embedding retriever** that finds candidate songs by meaning instead of exact label matches. A **reliability system** (guardrails, unit tests, and an evaluation harness) measures whether those additions improve on the original recommender.

It matters because the original had a documented weakness: exact-match labels gave `chill` and `relaxed` no credit for each other, and users had to speak the system's vocabulary. This version fixes that while keeping the ranking deterministic and every recommendation explainable. It also treats evaluation as a core feature rather than an afterthought.

## Architecture Overview

The full diagram is in [diagrams/architecture.md](diagrams/architecture.md). In short, data moves through five stages:

1. **Input and guardrails (implemented).** The user's request is cleaned before anything else happens: energy is clamped to 0-1, casing and whitespace are normalized, and unknown labels are flagged.
2. **Query parser (implemented; measured live, see Testing Summary).** `src/gemini_parser.py` sends the request to a Gemini Flash-Lite model (default `gemini-3.5-flash-lite`, override with the `GEMINI_MODEL` environment variable) with a JSON schema and converts the answer into a `UserProfile` (genre, mood, target energy, likes acoustic). Everything Gemini returns is validated against the catalog, so an invented genre or mood is discarded. If there are no credentials, the call fails, the response is incomplete, empty or off-schema, the keyword parser in `src/parser.py` takes over so the project still runs offline, and the fallback is noted in the output.
3. **Retriever (planned).** Local embeddings over song descriptions pull candidate songs by meaning. This is the RAG step: the system looks up information before answering.
4. **Core recommender (existing).** `score_song()` and `recommend_songs()` rank the candidates and produce a reason for each score. Ranking stays deterministic and is never delegated to the LLM.
5. **Output.** The top-k list with scores and reasons.

**Testing and human checks.** Unit tests (pytest) cover the deterministic parts and mock the Gemini parser. An evaluation harness runs the standard and adversarial profiles through the full pipeline, compares results to a golden set and to the non-RAG baseline, and writes a report. Humans stay in the loop at three points: reviewing the parsed profile and retrieved candidates, reading the recommendations and explanations, and reviewing the eval report before any weight or golden-set changes.

## Setup Instructions

1. **Clone the repository and enter it.**

   ```bash
   git clone <your-repo-url>
   cd applied-ai-system-final
   ```

2. **Create and activate a virtual environment** (recommended).

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

3. **Install dependencies.**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Set your Gemini API key** to use the Gemini parser (`--parser gemini`). Without a key the app falls back to the keyword parser and says so. Create a key in Google AI Studio. Live runs make real API calls to a Gemini model; check your plan's quota and pricing.

   ```bash
   export GEMINI_API_KEY="your-key-here"        # Mac or Linux
   $env:GEMINI_API_KEY = "your-key-here"        # Windows PowerShell
   ```

5. **Run the app.** With no arguments it runs the ten built-in profiles. Pass `--query` for a free-text request.

   ```bash
   python -m src.main
   python -m src.main --query "something mellow and acoustic to study to"
   python -m src.main --query "music to fall asleep to" --parser gemini
   ```

6. **Run the tests.**

   ```bash
   python -m pytest
   ```

7. **Run the evaluation harness.** It scores the pipeline against the human-judged golden set in `data/golden_set.json` and prints a report. Use `--out` to save it and `--strict` to exit non-zero on any failure.

   ```bash
   python -m src.evaluation
   python -m src.evaluation --out reports/baseline.md --label "my run"
   python -m src.evaluation --parser gemini --out reports/gemini.md   # live API calls
   python -m src.evaluation --parser gemini --runs 5 --out reports/gemini_5runs.md
   ```

   With `--parser gemini` the report counts how many answers silently fell back to the keyword parser, so a missing key can't be mistaken for a Gemini result.

   `--runs N` (1 to 20) repeats the whole evaluation and reports each metric as a mean with its range, plus how many runs each parser-dependent case passed. Use it for the Gemini parser, which is not deterministic. Each live run makes about 21 API calls.

## Sample Interactions

All three examples below are real output from the current code. Free-text parsing in these examples uses the offline keyword parser. Live Gemini results are in the Testing Summary, and the embedding retriever is still planned.

### Example 1: Structured profile (works today)

**Input:** `genre=lofi, mood=chill, energy=0.40, likes_acoustic=True`

```
1. Library Rain - Paper Lanterns        Score: 0.96 / 1.00
     * energy 0.35 vs target 0.40 (+0.38)
     * mood match (+0.25)
     * genre match (+0.20)
     * acoustic fit 0.86 (+0.13)

2. Midnight Coding - LoRoom             Score: 0.95 / 1.00
3. Spacewalk Thoughts - Orbit Bloom     Score: 0.74 / 1.00
```

### Example 2: Free-text request (works today with the keyword fallback parser)

**Input:** `python -m src.main --query "something mellow and acoustic to study to"`

```
'lofi' | 'chill' | energy 0.35 | likes acoustic

1. Library Rain - Paper Lanterns        Score: 0.98 / 1.00
   Why: energy 0.35 vs target 0.35 (+0.40), mood match (+0.25),
        genre match (+0.20), acoustic fit 0.86 (+0.13)
2. Midnight Coding - LoRoom             Score: 0.93 / 1.00
3. Spacewalk Thoughts - Orbit Bloom     Score: 0.76 / 1.00
```

The word "mellow" is not a label in the catalog. Without a parser it would score zero for mood; the keyword parser maps it to `chill`. The Gemini parser (`--parser gemini`) handles phrasings the keyword lists don't cover (see the Testing Summary), and the embedding retriever is planned.

### Example 3: Messy input and the guardrails (works today)

**Input:** genre `'Lofi '` (capital letter, trailing space), mood `'CHILL'`, energy 0.40

| | Genre match | Top score |
|---|---|---|
| **Before guardrails** | Lost, because the trailing space breaks the exact match | 0.76 |
| **With guardrails** | Restored after trimming whitespace | 0.96 |

The CLI now prints a warning for each fix:

```
! genre 'Lofi ' normalized to 'lofi'
! mood 'CHILL' normalized to 'chill'
```

Energy `1.50` is likewise clamped to 1.0 with the warning `energy 1.5 is out of range and was clamped to 1.0`.

## Design Decisions

- **Hybrid LLM and local retrieval.** Gemini handles the part where an LLM is clearly strongest: understanding vague language. Retrieval and ranking stay local, so they are free, offline, and reproducible. The trade-off is two moving parts instead of one.
- **The LLM never ranks.** Scoring remains the original transparent formula. This keeps explanations faithful to the real score and makes evaluation stable. The trade-off is that the system can't use the LLM's broader musical judgment.
- **Validate everything the LLM returns.** The parser's output is checked against the labels in the CSV, so a hallucinated genre or mood can't reach the scorer.
- **Graceful fallback.** A keyword parser means a missing API key or network failure degrades the experience instead of breaking it, and graders can run the project without credentials.
- **Evaluation is a first-class component.** Because the Gemini step is non-deterministic, the Gemini parser is mocked in unit tests and checked separately against a golden set of phrase-to-profile pairs.
- **Expanded catalog.** The original 20 songs, mostly one per genre, were too few for retrieval or precision@k to mean much. The catalog now has 80 songs with at least 3 per genre, four new genres (afrobeats, k-pop, bossa nova, blues) to reduce the Western-music bias noted in the model card, and a `description` column for the retriever. Data-quality tests in `tests/test_catalog.py` guard it. The songs are fictional, and 80 songs is still small, so results should not be read as evidence about real-world catalogs.

## Testing Summary

**What has been tested so far (original system).** Ten hand-written profiles (three standard, seven adversarial) were run through the scorer and the output was read by hand. On the original 20-song catalog, a mood-ablation experiment, where mood was commented out of the score, changed the recommendations for only four of the ten profiles. (This has not been re-run on the 80-song catalog.) This showed that mood is a weak signal and that energy and genre dominate the ranking.

**What worked.** The scorer is deterministic and every result comes with a reason, which made failures easy to spot. The adversarial profiles exposed real limitations.

**What didn't.**
- A trailing space in a profile label silently broke the genre match (see Example 3).
- Out-of-range energy (1.5) was accepted without complaint.
- Unknown genres and moods fell back to energy-only ranking with no warning.
- Review was manual, so nothing would catch a regression if the weights changed.

**Baseline evaluation (implemented).** `python -m src.evaluation` runs 22 ranking cases (16 profiles and 6 free-text queries parsed end to end) and 15 parser cases from a human-judged golden set. Each ranking case grades songs as *ideal* or *acceptable* and lists *forbidden* songs that must not appear in the top 5. A case passes only if the top result is relevant, no forbidden song appears, and an ideal song is in the top 3. I fixed these criteria before looking at results. The saved baseline is in [reports/baseline.md](reports/baseline.md):

| Metric | Baseline |
|---|---|
| Mean precision@5 | 0.73 |
| Mean nDCG@5 | 0.74 |
| Top-1 result relevant | 85% |
| Ranking cases passed | 19/22 |
| Forbidden songs recommended | 0 |
| Parser accuracy, easy phrasings | 10/10 |
| Parser accuracy, hard phrasings | 1/5 |
| Repeat runs identical (profile cases) | yes |

**An earlier, looser golden set hid the weaknesses.** My first version used one flat list of "relevant" songs per case and scored 100% top-1 on 8 scored cases. Grading relevance, adding forbidden songs, and adding end-to-end queries lowered the top-1 rate to 85% without changing the system at all, so the earlier number was measuring how generous my judgments were.

What this tells me: all 16 profile cases pass, so the scorer is doing its job when it gets a well-formed profile. All 3 failures are free-text queries the keyword parser misunderstands: "music to fall asleep to" (it only knows "sleep"), "songs that feel like a warm hug" (no cue words at all, so it falls back to a mid-energy default and recommends unrelated songs), and "something happy and danceable" (an ideal song misses the top 3). Those are the cases the Gemini parser should win, and the same golden set will measure whether it does. The ranking numbers on profile cases are already high, so I don't expect the embedding retriever to move them much. If it doesn't, that is a result worth reporting honestly.

Caveats: the golden set is still small (22 ranking and 15 parser cases), and I drafted all the judgments myself, so they reflect my taste. One judgment is arguable: "Sunday Brunch Swing" (a lively happy swing tune) is the keyword baseline's top answer to "something happy and danceable" and could reasonably count as acceptable. I left it out rather than adjust the golden set after seeing results. The 135 pytest tests also guard the metrics, the golden-set format and the baseline.

**Gemini parser tests (implemented).** 30 tests use a fake client, so no network or key is needed. They check the request shape (model, JSON schema, no sampling parameters), that invented labels and out-of-range energy are cleaned up, that user text is passed as data rather than instructions, and that each failure path (API error, incomplete or empty response, output that does not match the schema, missing credentials) falls back to the keyword parser. This tests my code around Gemini, not how well Gemini understands music requests.

**Gemini evaluation (live, one run).** `python -m src.evaluation --parser gemini` against `gemini-3.5-flash-lite`, with 0 of 21 answers falling back to the keyword parser. Saved in [reports/gemini.md](reports/gemini.md).

| Metric | Keyword baseline | Gemini parser |
|---|---|---|
| Ranking cases passed | 19/22 | 21/22 |
| Top-1 result relevant | 85% | 100% |
| Mean nDCG@5 | 0.74 | 0.85 |
| Mean precision@5 | 0.73 | 0.84 |
| Parser accuracy, easy phrasings | 10/10 | 9/10 |
| Parser accuracy, hard phrasings | 1/5 | 5/5 |
| Forbidden songs recommended | 0 | 0 |

The hard free-text phrasings the keyword parser missed ("music to fall asleep to", "background music while I work", "something happy and danceable") now parse correctly, and the end-to-end ranking improves with them. The profile cases don't use the parser, so they are unchanged.

Read these numbers with care:
- **One run is a noisy measurement.** Gemini is not deterministic. For "something mellow and acoustic to study to", three repeat calls returned genre `lofi` once and an empty genre twice. That phrase is the one easy case Gemini failed (the keyword parser passes it because I hard-coded "study" as a lofi cue, and the golden set expects `lofi`, so both reflect my judgment that studying implies lofi). The 14/15 vs 11/15 parser gap is three phrases, so it could shrink or grow on another run. Repeated runs are needed before claiming a precise improvement.
- **The golden set is small and mine.** Its judgments, including the arguable ones noted above, shaped both the baseline and these results.
- **The remaining ranking failure** is "songs that feel like a warm hug": Gemini parsed it sensibly, but an ideal song did not reach the top 3 (it returned relaxed, acoustic songs such as Coffee Shop Stories that I graded only as acceptable).

**Still to do.** Run `python -m src.evaluation --parser gemini --runs 5` to measure run-to-run variance, then decide whether the embedding retriever is worth adding. The ranking numbers on profile cases are already high, so I expect little from it, and that would be worth reporting honestly.

**What I learned so far.** Hand-reading output doesn't scale, and the failures worth catching are the quiet ones: results that look plausible but are wrong. That is the main reason the testing system is part of the plan.

> Fill in the real pass/fail counts and metrics here after the harness runs.

## Reflection

The original project taught me that a recommender is a set of design choices: which features to score, how to weight them, and who gets left out. Extending it taught me that adding an AI component, like an LLM or a retriever, doesn't remove those choices. It moves them. Using Gemini for language and keeping the ranking deterministic was a decision about where to trust a model and where to keep control. Planning the evaluation before writing the features made the project's goal clearer: the question isn't whether the new version feels smarter, it's whether it measurably does better than the baseline.

> Revise this section in your own words after the implementation is finished.

## Project Files

- [diagrams/architecture.md](diagrams/architecture.md): system architecture diagram
- [model_card.md](model_card.md): model card for the original recommender
- [ai_interactions.md](ai_interactions.md): log of AI interactions
- `src/`: recommender code
- `data/songs.csv`: song catalog
- `tests/`: pytest tests
