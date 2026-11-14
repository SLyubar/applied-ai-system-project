# System Architecture

This diagram shows the existing content-based recommender (grey) and the two planned enhancements: a **RAG retrieval layer** (blue) and a **reliability / testing system** (orange). Human checkpoints are shown in green.

```mermaid
flowchart TD
    %% ---------- INPUT ----------
    subgraph INPUT["1. Input"]
        U["User<br/>free-text request or structured profile"]
        CSV[("data/songs.csv<br/>20 songs")]
        KB[("Song descriptions<br/>embedded once, cached in cache/")]
    end

    %% ---------- RAG ----------
    subgraph RAG["2. Retrieval (NEW - RAG)"]
        QP["Query Parser - Gemini API<br/>text to UserProfile<br/>validated against known labels<br/>fallback: keyword parser if no key or API error"]
        RET["Retriever<br/>Gemini embeddings (semantic)<br/>or local TF-IDF (offline, word match)<br/>searches descriptions by the raw query"]
        CAND["Candidate Songs<br/>top 20 by default<br/>shown to the user for review"]
    end

    %% ---------- CORE ----------
    subgraph CORE["3. Core Recommender (existing)"]
        LOAD["load_songs()<br/>parse CSV"]
        SCORE["score_song()<br/>0.40 energy + 0.25 mood<br/>+ 0.20 genre + 0.15 acoustic"]
        RANK["recommend_songs()<br/>sort and take top k"]
        EXPL["Explanation Builder<br/>per-feature reasons"]
    end

    %% ---------- OUTPUT ----------
    subgraph OUTPUT["4. Output"]
        OUT["Top-k Recommendations<br/>title, artist, score, why"]
    end

    %% ---------- RELIABILITY ----------
    subgraph REL["5. Reliability and Testing (NEW)"]
        GUARD["Input Guardrails<br/>clamp energy 0-1<br/>normalize case and whitespace<br/>flag unknown labels"]
        UNIT["Unit Tests - pytest + CI<br/>264 tests, 99 percent coverage<br/>Gemini parser mocked"]
        EVAL["Evaluation Harness<br/>22 ranking + 15 parser cases<br/>precision, nDCG, forbidden songs<br/>repeated runs with mean and range"]
        ROB["Robustness Checks<br/>paraphrase consistency<br/>injection, non-English,<br/>gibberish, long input"]
        ABL["Ablation Study<br/>drop or nudge each weight"]
        GOLD[("Golden Set<br/>graded ideal and acceptable songs<br/>forbidden songs, paraphrase groups,<br/>adversarial cases")]
        REPORT["Eval Reports<br/>reports/*.md"]
        LOG[("Interaction Log and ratings<br/>logs/interactions.jsonl")]
    end

    %% ---------- HUMAN ----------
    subgraph HUMAN["Human in the loop"]
        H1{{"Review parsed profile<br/>and retrieved candidates"}}
        H2{{"Review recommendations<br/>and explanations"}}
        H3{{"Review eval report<br/>tune weights, fix failures"}}
    end

    %% ---------- MAIN DATA FLOW ----------
    U --> GUARD
    GUARD --> QP
    CSV --> LOAD
    CSV --> KB
    KB --> RET
    GUARD -. "raw query text" .-> RET
    RET --> CAND
    QP -. "structured profile" .-> SCORE
    CAND --> SCORE
    LOAD --> SCORE
    SCORE --> RANK
    RANK --> EXPL
    EXPL --> OUT

    %% ---------- TESTING FLOW ----------
    UNIT -. "tests" .-> SCORE
    UNIT -. "tests" .-> GUARD
    GOLD --> EVAL
    EVAL -. "runs profiles through" .-> GUARD
    OUT -. "captured results" .-> EVAL
    EVAL --> REPORT
    GOLD --> ROB
    ROB -. "parses queries" .-> QP
    ROB --> REPORT
    GOLD --> ABL
    ABL -. "re-scores with other weights" .-> SCORE
    ABL --> REPORT
    OUT --> LOG

    %% ---------- HUMAN CHECKPOINTS ----------
    QP --> H1
    CAND --> H1
    H1 -. "correct or approve" .-> QP
    OUT --> H2
    H2 -. "feedback" .-> LOG
    REPORT --> H3
    H3 -. "update weights and golden set" .-> SCORE
    H3 -. "update golden set" .-> GOLD
    LOG -. "bad ratings export as candidate cases" .-> H3

    %% ---------- STYLES ----------
    classDef existing fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef rag fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef rel fill:#ffedd5,stroke:#ea580c,color:#111827
    classDef human fill:#dcfce7,stroke:#16a34a,color:#111827
    classDef data fill:#f3f4f6,stroke:#9ca3af,color:#111827

    class LOAD,SCORE,RANK,EXPL,OUT,U existing
    class QP,RET,CAND rag
    class GUARD,UNIT,EVAL,ROB,ABL,REPORT rel
    class H1,H2,H3 human
    class CSV,KB,GOLD,LOG data
```

## How to read it

**Data flow (input, process, output)**
1. The user's request passes through the input guardrails, which clamp and normalize it.
2. Two things happen in parallel. The Gemini-powered query parser turns the request into a `UserProfile`, checked against the known genre and mood labels (a keyword parser takes over if there is no key or the call fails). Separately, the retriever embeds the raw query text and finds the 20 songs whose descriptions are closest in meaning.
3. The existing `score_song()` and `recommend_songs()` rank only those candidates. Scoring stays deterministic and explainable. Without a query (a ready-made profile), the whole catalog is ranked.
4. The output is the top-k list with a reason for each score, and the retrieved candidates are shown for review.

**Testing**
- **Unit tests** (pytest, run in CI) cover guardrails, scoring, both parsers, the retriever and the evaluation code itself. The Gemini calls are faked, so no key is needed.
- The **evaluation harness** scores ranking quality against a golden set with graded relevance and forbidden songs, and repeats runs to show variance for the non-deterministic parser. The **robustness checks** test paraphrases and adversarial input, and the **ablation study** re-scores with different weights. Reports are written to `reports/`.
- The retrieval experiment compares embedders and candidate-pool sizes on the free-text cases.

**Human checkpoints**
- **H1** checks that the parsed profile and retrieved candidates make sense before ranking, since this is where RAG can go wrong.
- **H2** is the user or developer reading the recommendations and explanations. `python -m src.feedback` logs a good or bad rating, and bad ratings export as skeleton golden-set cases.
- **H3** reviews the eval report and decides on weight changes or golden-set updates. Nothing changes automatically.
