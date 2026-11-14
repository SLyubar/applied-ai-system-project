# 🎵 Music Recommender, Extended with RAG and a Reliability System

> **Status.** Everything described here is implemented and tested offline. Live Gemini results come from earlier runs, and the final live run could not finish because the free-tier daily quota ran out. The Testing Summary says exactly which numbers were measured when, and on which version of the test set.

## Original Project: Music Recommender Simulation

This project extends **Music Recommender Simulation**, a small CLI-first content-based music recommender. The original goal was to represent songs and a listener's taste profile as data, turn them into a transparent scoring rule, and evaluate where that rule succeeds and fails. It scored each song in a 20-song CSV on energy closeness, mood match, genre match, and acoustic preference (`0.40 / 0.25 / 0.20 / 0.15`), then returns the top-k songs with a plain-language reason for every point awarded. It was stress-tested against three standard and seven adversarial hand-written profiles.

## Title and Summary

**Music Recommender: Retrieval-Augmented Edition**

This project lets a listener ask for music in plain language ("something calm and acoustic for a rainy study session") and get ranked, explained recommendations. Three additions make that possible: a **Gemini-powered query parser** that turns free text into a structured taste profile, an **embedding retriever** that finds candidate songs by meaning instead of exact label matches, and a **confidence score** that says how much to trust each result. A **reliability system** (guardrails, automated tests, an evaluation harness with robustness checks, an ablation study, and a feedback log) measures whether those additions improve on the original recommender.

It matters because the original had a documented weakness: exact-match labels gave `chill` and `relaxed` no credit for each other, and users had to speak the system's vocabulary. This version fixes that while keeping the ranking deterministic and every recommendation explainable. It also treats evaluation as a core feature rather than an afterthought.

## Architecture Overview

The full diagram is in [diagrams/architecture.md](diagrams/architecture.md). Data moves through six stages:

1. **Input and guardrails.** The request is cleaned first: energy is clamped to 0-1, casing and whitespace are normalized, and unknown labels are flagged.
2. **Query parser.** `src/gemini_parser.py` sends the request to a Gemini Flash-Lite model (default `gemini-3.5-flash-lite`, override with `GEMINI_MODEL`) with a JSON schema and converts the answer into a profile (genre, mood, target energy, likes acoustic, plus the model's own certainty). Everything Gemini returns is validated against the catalog, so an invented genre or mood is discarded. If there are no credentials, the call fails or times out, or the response is incomplete or off-schema, the keyword parser in `src/parser.py` takes over and the output says why.
3. **Retriever (the RAG step).** `src/retrieval.py` embeds each song's description once (Gemini `gemini-embedding-001`, cached in `cache/`, or a local TF-IDF alternative) and finds the 20 songs closest in meaning to the raw request. The candidates are printed so a person can check them.
4. **Core recommender.** `score_song()` and `recommend_songs()` rank only those candidates and explain every point awarded. Ranking is deterministic and never delegated to an LLM.
5. **Confidence.** `src/confidence.py` combines how much of the request was understood with how well the best song fits it, and lists the reasons when it is not high.
6. **Output and log.** The top-k list with scores, reasons and confidence. Each `--query` run is logged so a person can rate it.

**Testing and human checks.** 382 pytest tests cover the deterministic parts and replace Gemini with a fake client. An evaluation harness scores ranking and parsing against a golden set, repeats runs to show the variance of the non-deterministic parser, and checks whether the confidence score means anything. Humans stay in the loop at three points: reviewing the retrieved candidates, rating results with `python -m src.feedback`, and reviewing reports before any weight or golden-set change.

## Setup Instructions

1. **Clone the repository and enter it.**

   ```bash
   git clone https://github.com/SLyubar/applied-ai-system-project
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

4. **(Optional) Set your Gemini API key** to use the Gemini parser and Gemini embeddings. Without a key the app falls back to the keyword parser and TF-IDF retrieval and says so. Create a key in Google AI Studio.

   ```bash
   export GEMINI_API_KEY="your-key-here"        # Mac or Linux
   $env:GEMINI_API_KEY = "your-key-here"        # Windows PowerShell
   ```

   Live calls count against your plan. When this was written the free tier allowed 500 requests per model per day and 100 embedding requests per minute, and the daily limit is shared with every other run that day. Evaluation runs print how many calls they are about to make.

5. **Run the app.** With no arguments it runs the ten built-in profiles. Pass `--query` for a free-text request.

   ```bash
   python -m src.main
   python -m src.main --query "something mellow and acoustic to study to"
   python -m src.main --query "slow dance at a wedding" --parser gemini --retriever gemini
   ```

   `--retriever tfidf|gemini` retrieves candidates by description before scoring (`--pool N` sets how many). `--no-log` skips the interaction log.

6. **Run the tests.**

   ```bash
   python -m pytest
   python -m pytest --cov=src          # with coverage
   ```

7. **Run the evaluation harness.** It scores the pipeline against the golden set in `data/golden_set.json` and writes a report. `--out` saves it, `--strict` exits non-zero on any failure, `--runs N` repeats it (1 to 20) and reports each metric as a mean with its range.

   ```bash
   python -m src.evaluation
   python -m src.evaluation --out reports/baseline.md
   python -m src.evaluation --parser gemini --runs 3 --out reports/gemini_3runs.md      # live calls
   python -m src.evaluation --parser gemini --runs 3 --retriever gemini --pool 20       # live calls
   ```

   With `--parser gemini` the report counts how many answers silently fell back to the keyword parser, so a missing key or exhausted quota can't be mistaken for a Gemini result.

8. **Run the experiments.**

   ```bash
   python -m src.retrieval_eval --parser gemini --embedders tfidf,gemini --out reports/retrieval.md
   python -m src.ablation --out reports/ablation.md
   ```

9. **Rate results and review them.** Every `--query` run prints an id.

   ```bash
   python -m src.feedback list
   python -m src.feedback rate <id> good        # or bad, optionally --note "why"
   python -m src.feedback export-candidates     # skeleton golden-set cases for runs rated bad
   ```

## Sample Interactions

The examples below are real output. Examples 1 to 3 and 5 use the offline keyword parser; example 4 was captured from a live Gemini run before the confidence line was added.

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

The word "mellow" is not a label in the catalog. Without a parser it would score zero for mood; the keyword parser maps it to `chill`. The Gemini parser (`--parser gemini`) handles phrasings the keyword lists don't cover (see the Testing Summary).

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

### Example 4: Retrieval fixes a request the keyword parser misreads

**Input:** `"slow dance at a wedding"`

The keyword parser reads "dance" as high energy and recommends EDM: *Afterhours Static, Sunrise Drop, Pulse Garden*. The local TF-IDF retriever matches words, not meaning: its candidates mix the right song (*Last Call Ballad*) with *Calle Fuego*, *Noche de Salsa* and *Gym Hero*, and the misread profile still wins, so the top recommendation is a metal song, *Black Lantern Hymn*. With Gemini parsing and Gemini embeddings (live):

```
'pop' | 'romantic' | energy 0.35 | no acoustic preference
! retrieved 20 candidates (gemini); closest: Last Call Ballad, Velvet Hours, Slow Dance Voicemail,
  Midnight in Accra, Letters to Orchard Lane

1. Paper Hearts - Maple June              Score: 0.77 / 1.00
2. Letters to Orchard Lane - Fern & Fable Score: 0.64 / 1.00
3. Last Call Ballad - Marlowe Quintet     Score: 0.63 / 1.00
```

### Example 5: Confidence says when not to trust a result

A clear profile:

```
PROFILE: Starter lofi
  Confidence: HIGH (0.96)  [understanding 1.00, match 0.96]  matched on: energy, mood, genre, acoustic
```

Labels the catalog has never heard of (this profile used to print a confident-looking score with no warning):

```
PROFILE: Unknown genre and mood
'vaporwave' | 'sad' | energy 0.50 | prefers non-acoustic
  Confidence: LOW (0.55)  [understanding 0.55, match 0.93]  matched on: energy, acoustic
    - genre 'vaporwave' is not in the catalog
    - mood 'sad' is not in the catalog
  Low confidence: treat these as rough suggestions, not a good match.
```

Gibberish:

```
python -m src.main --query "asdkjh qwpoeiu zzzz"
  Confidence: LOW (0.00)  [understanding 0.00, match 0.00]  matched on: nothing
    - nothing in the request could be turned into a criterion
  Low confidence: treat these as rough suggestions, not a good match.
```

## Design Decisions

- **Hybrid LLM and local retrieval.** Gemini handles the part where an LLM is clearly strongest: understanding vague language. Retrieval and ranking stay local, so they are free, offline, and reproducible. The trade-off is two moving parts instead of one.
- **The LLM never ranks.** Scoring remains the original transparent formula. This keeps explanations faithful to the real score and makes evaluation stable. The trade-off is that the system can't use the LLM's broader musical judgment.
- **Validate everything the LLM returns.** The parser's output is checked against the labels in the CSV, so a hallucinated genre or mood can't reach the scorer.
- **Graceful fallback.** A keyword parser means a missing API key or network failure degrades the experience instead of breaking it, and graders can run the project without credentials.
- **Evaluation is a first-class component.** Because the Gemini step is non-deterministic, the Gemini parser is mocked in unit tests and checked separately against a golden set of phrase-to-profile pairs.
- **Expanded catalog.** The original 20 songs, mostly one per genre, were too few for retrieval or precision@k to mean much. The catalog now has 80 songs with at least 3 per genre, four new genres (afrobeats, k-pop, bossa nova, blues) to reduce the Western-music bias noted in the model card, and a `description` column for the retriever. Data-quality tests in `tests/test_catalog.py` guard it. The songs are fictional, and 80 songs is still small, so results should not be read as evidence about real-world catalogs.

- **Retrieve, then score.** Retrieval picks candidates from the request's raw text; the unchanged scorer ranks them. This widens what a request can match without changing how songs are scored or explained. The trade-off, measured below, is that a pool that is too small can drop the right song, and for free-text requests retrieval alone turned out to be nearly as good as the full pipeline.
- **Evidence over assumption for embeddings.** I built a local TF-IDF embedder and a Gemini embedder behind one interface and compared them. TF-IDF made results worse (it matches words, not meaning), so Gemini embeddings are the recommended option; TF-IDF stays as the offline fallback.
- **Confidence is explainable, not a probability.** `min(understanding, match)`, banded high (0.8 or more), medium (0.6 or more) and low, with reasons. An LLM's self-reported certainty can only lower it, because models tend to overstate how sure they are. Whether the score separates good results from bad is measured, not assumed (see calibration).
- **Fail fast, and say why.** A used-up daily quota once looked like a hang because the SDK retried for hours. Only transient server errors are retried now, every call has a timeout, and any failure falls back with the reason in the output.
- **No automatic learning from feedback.** Ratings are logged and bad ones export as skeleton test cases, but a person must fill in the right answers. Nothing changes the golden set or the weights on its own.

## Testing Summary

**In short.** 382 automated tests pass (99% line coverage). Against a hand-built golden set of 32 ranking cases and 15 parser phrases, the keyword baseline passes 21/32 ranking cases and 11/15 phrases, and fails every non-English request. In live runs on an earlier version of the golden set, the Gemini parser got 14.2 of 15 phrases right (range 13 to 15 over 5 runs) and passed 12 of 12 adversarial cases in all 3 runs, including both prompt injections and four non-English requests. Gemini-embedding retrieval raised free-text nDCG from 0.65 to 0.69-0.73, while TF-IDF retrieval lowered it to 0.54. The system struggled when a phrase stated no energy, and with vague or figurative requests. The confidence score reliably flags unknown labels and gibberish, but it cannot see a confident misreading. The final live run on the finished code could not complete because the free-tier daily quota ran out.

### How reliability is measured

| Method | What it covers |
|---|---|
| Automated tests | 382 pytest tests for guardrails, scoring, both parsers, the retriever, confidence, the CLI, the feedback log, and the evaluation code itself. Gemini is replaced by a fake client, including the real error classes it raises. A GitHub Actions workflow runs them (not yet exercised, since it needs a push). |
| Evaluation harness | Ranking cases with graded relevance (ideal and acceptable songs) and forbidden songs; a case passes only if the top result is relevant, nothing forbidden appears, and an ideal song is in the top 3. Criteria were fixed before looking at results. Parser phrases, repeated runs with mean and range, and fallback detection so a keyword answer can't pass as a Gemini one. |
| Robustness checks | 6 paraphrase groups (every wording must meet the expectation and agree on energy) and 12 adversarial cases (injection, other languages, gibberish, off-topic, very long input). |
| Ablation | Drops or nudges each scoring weight and re-scores the profile cases. |
| Confidence scoring | Per result, with a calibration table in every report. |
| Logging and error handling | Every failure path falls back and records why, for example `RateLimitError 429: ... 500 requests per day`. `--query` runs are logged with their profile, notes, results and confidence. |
| Human evaluation | `python -m src.feedback` records good or bad ratings. **No ratings have been collected yet**, so this method is built but unused. |

### Results

**Keyword baseline, current golden set** ([reports/baseline.md](reports/baseline.md)):

| Metric | Value |
|---|---|
| Ranking cases passed | 21/32 |
| Top-1 result relevant | 73% |
| Mean nDCG@5 | 0.61 |
| Parser phrases passed (easy / hard) | 11/15 (100% / 20%) |
| Paraphrase groups passed | 0/6 |
| Adversarial cases passed | 8/12 (fails all four non-English requests) |
| Forbidden songs recommended | 2 |

All 16 profile cases pass, so the scorer works when it gets a well-formed profile. The failures are free-text requests the keyword parser misreads, for example "dance" in "slow dance at a wedding" read as high energy.

**Gemini parser, live** ([5 runs](reports/gemini_5runs.md) and [3 runs with robustness](reports/gemini_full_3runs.md)). These ran on an earlier, smaller golden set (22 ranking cases, before the 10 extra free-text cases and the confidence code), with 0 answers falling back to the keyword parser:

| | Keyword baseline (same set) | Gemini, 5 runs: mean (min to max) |
|---|---|---|
| Ranking cases passed (of 22) | 19 | 21.2 (21 to 22) |
| Mean nDCG@5 | 0.74 | 0.84 (0.81 to 0.85) |
| Parser phrases passed (of 15) | 11 | 14.2 (13 to 15) |
| Hard phrasings | 20% | 96% (80% to 100%) |
| Easy phrasings | 100% | 94% |
| Adversarial cases passed | 8/12 | 12/12 in all 3 runs |
| Paraphrase groups passed | 0/6 | 5/6 |

Even Gemini's worst run beat the baseline. Two findings:
- **The "Sad" paraphrase group failed (0 of 3 runs), but Gemini was not wrong.** It gave mood `melancholy` for all four wordings. "I feel sad tonight" and "something melancholy" state no energy, so they got the default 0.5, while my expectation required 0.45 or lower. That is a weak expectation on my part, and I left it unchanged because adjusting it after seeing results would be tuning.
- **The study phrase is unstable.** "Something mellow and acoustic to study to" passed only 2 of 5 runs, because Gemini sometimes returns no genre. The golden set expects `lofi`, which reflects my judgment that studying implies lofi.

**Retrieval experiment** ([reports/retrieval.md](reports/retrieval.md)): Gemini parser, 16 free-text cases, every configuration reusing the same parsed profile for each query.

| Configuration | Cases passed | nDCG@5 |
|---|---|---|
| No retrieval | 11/16 | 0.65 |
| TF-IDF, retrieve 20 then score | 8/16 | 0.54 |
| Gemini embeddings, retrieve 10 then score | 13/16 | 0.73 |
| Gemini embeddings, retrieve 20 then score | 14/16 | 0.69 |
| Gemini embeddings, retrieve 40 then score | 12/16 | 0.68 |
| Gemini embeddings, retrieval only (no scorer) | 15/16 | 0.70 |

Gemini embeddings help; TF-IDF hurts. The surprise is that retrieval alone passes the most cases, so for free-text requests the scorer adds little. The pool-size differences are within noise. I wrote both the song descriptions and the test queries, so their vocabulary overlaps in a way that favors retrieval, and 16 cases is a small sample.

**Ablation** ([reports/ablation.md](reports/ablation.md)): dropping genre hurts most (nDCG 0.87 to 0.69), dropping mood least (to 0.83). The weights I chose are not the best on my own golden set (halving mood scores 0.95, equal weights 0.91), but tuning them to judgments I wrote myself would be overfitting, so I did not change them.

**Confidence calibration** (keyword parser, current golden set; every report has this table):

| Confidence | Cases | Passed | Pass rate |
|---|---|---|---|
| high | 22 | 15 | 68% |
| medium | 4 | 4 | 100% |
| low | 4 | 0 | 0% |

Every low-confidence case failed, so a low rating is a reliable warning. A high rating is not a guarantee: the parser's understanding score was 1.00 on phrases it got right but 0.25 on ones it got wrong, yet it still rates a confident misreading like "slow dance" as EDM highly. Gemini's self-reported certainty is meant to catch that, but **that has not been measured live yet.**

### What worked, what didn't, what I learned

- **A looser golden set hid the weaknesses.** My first version scored 100% top-1 on the profile cases. With graded relevance and forbidden songs the same system scored 85%, and 73% once the harder free-text cases were added. The system never changed; the tests got honest.
- **Testing found real bugs in my own work.** A loophole where a default energy of 0.5 satisfied "calm" expectations; Gemini's Interactions API raising a different error family than the parser caught (a timeout would have crashed it, and my tests used the wrong class so they never noticed); and a daily quota that the SDK silently retried for hours, which looked like a hang.
- **The LLM helped most where it was supposed to**, on vague phrasing and other languages, and did not break on injection. It is not deterministic, so one run is not a measurement.
- **TF-IDF retrieval was a bad idea that only measuring revealed.**

### Known gaps

- **The final live run is missing.** The free tier's daily limit for `gemini-3.5-flash-lite` (500 requests) was used up, and the API said to retry in about 15 hours. Not yet measured live on the finished code: Gemini with the 10 extra free-text cases, Gemini's self-reported confidence and its calibration, and the end-to-end pipeline with Gemini retrieval over several runs. The commands are in Setup, steps 7 and 8; they need about 200 calls per 3 runs.
- **One author.** I drafted every golden-set judgment and the descriptions, so results reflect one person's taste; there is no second reviewer, and no human ratings have been collected.
- **Small samples.** 32 ranking cases, 15 parser phrases, 16 free-text retrieval cases.
- **Confidence cannot see a confident mistake**, as shown above.
- **The GitHub Actions workflow has not run**, and cost is not tracked (only wall time).
- **The catalog is 80 fictional songs.** Nothing here says how this would behave on a real catalog.

## Reflection

The original project taught me that a recommender is a set of design choices: which features to score, how to weight them, and who gets left out. Extending it taught me that adding an AI component, like an LLM or a retriever, doesn't remove those choices. It moves them. Using Gemini for language and keeping the ranking deterministic was a decision about where to trust a model and where to keep control. Planning the evaluation before writing the features made the project's goal clearer: the question isn't whether the new version feels smarter, it's whether it measurably does better than the baseline.

> Revise this section in your own words after the implementation is finished.

## Project Files

- [diagrams/architecture.md](diagrams/architecture.md): system architecture diagram
- [model_card.md](model_card.md): model card for the original recommender
- [ai_interactions.md](ai_interactions.md): log of AI interactions
- `app.py`: Streamlit interface (`src/pipeline.py` is its query flow, `src/golden_writer.py` saves feedback to the golden set)
- `src/recommender.py`, `src/guardrails.py`, `src/parser.py`: scoring, input cleaning, keyword parser
- `src/gemini_parser.py`, `src/retrieval.py`, `src/confidence.py`: Gemini parser, embedding retriever, confidence scoring
- `src/evaluation.py`, `src/robustness.py`, `src/ablation.py`, `src/retrieval_eval.py`: evaluation harness and experiments
- `src/feedback.py`, `src/main.py`: interaction log and ratings, command line runner
- `data/songs.csv`, `data/golden_set.json`: song catalog, human-judged test set
- `reports/`: saved evaluation reports (each states which version of the code and test set produced it)
- `tests/`: pytest tests
