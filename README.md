# 🎵 Music Recommender, Extended with RAG and a Reliability System

> **DRAFT.** This README describes the planned system. Sections marked **(planned)** describe features that are designed but not yet implemented. Update or remove those markers as each feature lands.

## Original Project: Music Recommender Simulation

This project extends **Music Recommender Simulation**, a small CLI-first content-based music recommender. The original goal was to represent songs and a listener's taste profile as data, turn them into a transparent scoring rule, and evaluate where that rule succeeds and fails. It scored each song in a 20-song CSV on energy closeness, mood match, genre match, and acoustic preference (`0.40 / 0.25 / 0.20 / 0.15`), then returns the top-k songs with a plain-language reason for every point awarded. It was stress-tested against three standard and seven adversarial hand-written profiles.

## Title and Summary

**Music Recommender: Retrieval-Augmented Edition**

This project lets a listener ask for music in plain language ("something calm and acoustic for a rainy study session") and get ranked, explained recommendations. Two additions make that possible: a **Claude-powered query parser** that turns free text into a structured taste profile, and a **local embedding retriever** that finds candidate songs by meaning instead of exact label matches. A **reliability system** (guardrails, unit tests, and an evaluation harness) measures whether those additions improve on the original recommender.

It matters because the original had a documented weakness: exact-match labels gave `chill` and `relaxed` no credit for each other, and users had to speak the system's vocabulary. This version fixes that while keeping the ranking deterministic and every recommendation explainable. It also treats evaluation as a core feature rather than an afterthought.

## Architecture Overview

The full diagram is in [diagrams/architecture.md](diagrams/architecture.md). In short, data moves through five stages:

1. **Input and guardrails (implemented).** The user's request is cleaned before anything else happens: energy is clamped to 0-1, casing and whitespace are normalized, and unknown labels are flagged.
2. **Query parser (keyword fallback implemented, Claude parser planned).** The Claude API will convert free text into a `UserProfile` (genre, mood, target energy, likes acoustic) and validate it against the labels in the catalog. If there is no API key or the call fails, the keyword parser in `src/parser.py` takes over so the project still runs offline. Today only the keyword parser exists.
3. **Retriever (planned).** Local embeddings over song descriptions pull candidate songs by meaning. This is the RAG step: the system looks up information before answering.
4. **Core recommender (existing).** `score_song()` and `recommend_songs()` rank the candidates and produce a reason for each score. Ranking stays deterministic and is never delegated to the LLM.
5. **Output.** The top-k list with scores and reasons.

**Testing and human checks.** Unit tests (pytest) cover the deterministic parts and mock the Claude parser. An evaluation harness runs the standard and adversarial profiles through the full pipeline, compares results to a golden set and to the non-RAG baseline, and writes a report. Humans stay in the loop at three points: reviewing the parsed profile and retrieved candidates, reading the recommendations and explanations, and reviewing the eval report before any weight or golden-set changes.

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

4. **(Planned) Set your Claude API key** to enable free-text parsing. This step is optional: without a key the app falls back to the keyword parser.

   ```bash
   export ANTHROPIC_API_KEY="your-key-here"     # Mac or Linux
   $env:ANTHROPIC_API_KEY = "your-key-here"     # Windows PowerShell
   ```

5. **Run the app.** With no arguments it runs the ten built-in profiles. Pass `--query` for a free-text request.

   ```bash
   python -m src.main
   python -m src.main --query "something mellow and acoustic to study to"
   ```

6. **Run the tests.**

   ```bash
   python -m pytest
   ```

7. **(Planned) Run the evaluation harness.** The exact command will be added when the harness is built.

## Sample Interactions

All three examples below are real output from the current code. Free-text parsing currently uses the offline keyword parser; the Claude parser and embedding retriever are still planned.

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

The word "mellow" is not a label in the catalog. Without a parser it would score zero for mood; the keyword parser maps it to `chill`. The Claude parser and embedding retriever (planned) will handle phrasings the keyword lists don't cover.

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

- **Hybrid LLM and local retrieval.** Claude handles the part where an LLM is clearly strongest: understanding vague language. Retrieval and ranking stay local, so they are free, offline, and reproducible. The trade-off is two moving parts instead of one.
- **The LLM never ranks.** Scoring remains the original transparent formula. This keeps explanations faithful to the real score and makes evaluation stable. The trade-off is that the system can't use the LLM's broader musical judgment.
- **Validate everything the LLM returns.** The parser's output is checked against the labels in the CSV, so a hallucinated genre or mood can't reach the scorer.
- **Graceful fallback.** A keyword parser means a missing API key or network failure degrades the experience instead of breaking it, and graders can run the project without credentials.
- **Evaluation is a first-class component.** Because the Claude step is non-deterministic, the Claude parser is mocked in unit tests and checked separately against a golden set of phrase-to-profile pairs.
- **Expanded catalog.** The original 20 songs, mostly one per genre, were too few for retrieval or precision@k to mean much. The catalog now has 80 songs with at least 3 per genre, four new genres (afrobeats, k-pop, bossa nova, blues) to reduce the Western-music bias noted in the model card, and a `description` column for the retriever. Data-quality tests in `tests/test_catalog.py` guard it. The songs are fictional, and 80 songs is still small, so results should not be read as evidence about real-world catalogs.

## Testing Summary

**What has been tested so far (original system).** Ten hand-written profiles (three standard, seven adversarial) were run through the scorer and the output was read by hand. On the original 20-song catalog, a mood-ablation experiment, where mood was commented out of the score, changed the recommendations for only four of the ten profiles. (This has not been re-run on the 80-song catalog.) This showed that mood is a weak signal and that energy and genre dominate the ranking.

**What worked.** The scorer is deterministic and every result comes with a reason, which made failures easy to spot. The adversarial profiles exposed real limitations.

**What didn't.**
- A trailing space in a profile label silently broke the genre match (see Example 3).
- Out-of-range energy (1.5) was accepted without complaint.
- Unknown genres and moods fell back to energy-only ranking with no warning.
- Review was manual, so nothing would catch a regression if the weights changed.

**What the new system will add (planned).** Pytest tests for score bounds and ranking determinism, guardrail tests for clamping and normalization, a golden-set evaluation reporting precision@k and consistency, and a side-by-side comparison of the original recommender against the RAG version.

**What I learned so far.** Hand-reading output doesn't scale, and the failures worth catching are the quiet ones: results that look plausible but are wrong. That is the main reason the testing system is part of the plan.

> Fill in the real pass/fail counts and metrics here after the harness runs.

## Reflection

The original project taught me that a recommender is a set of design choices: which features to score, how to weight them, and who gets left out. Extending it taught me that adding an AI component, like an LLM or a retriever, doesn't remove those choices. It moves them. Using Claude for language and keeping the ranking deterministic was a decision about where to trust a model and where to keep control. Planning the evaluation before writing the features made the project's goal clearer: the question isn't whether the new version feels smarter, it's whether it measurably does better than the baseline.

> Revise this section in your own words after the implementation is finished.

## Project Files

- [diagrams/architecture.md](diagrams/architecture.md): system architecture diagram
- [model_card.md](model_card.md): model card for the original recommender
- [ai_interactions.md](ai_interactions.md): log of AI interactions
- `src/`: recommender code
- `data/songs.csv`: song catalog
- `tests/`: pytest tests
