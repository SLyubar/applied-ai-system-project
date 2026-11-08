# System Architecture

This diagram shows the existing content-based recommender (grey) and the two planned enhancements: a **RAG retrieval layer** (blue) and a **reliability / testing system** (orange). Human checkpoints are shown in green.

```mermaid
flowchart TD
    %% ---------- INPUT ----------
    subgraph INPUT["1. Input"]
        U["User<br/>free-text request or structured profile"]
        CSV[("data/songs.csv<br/>20 songs")]
        KB[("Song descriptions / tags<br/>retrieval index")]
    end

    %% ---------- RAG ----------
    subgraph RAG["2. Retrieval (NEW - RAG)"]
        QP["Query Parser - Claude API<br/>text to UserProfile<br/>validated against known labels<br/>fallback: keyword parser if no key or API error"]
        RET["Retriever - local embeddings<br/>over song descriptions<br/>deterministic, offline, free"]
        CAND["Candidate Songs<br/>top N from index"]
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
        UNIT["Unit Tests - pytest<br/>score bounds 0 to 1<br/>deterministic ranking<br/>Claude parser mocked"]
        EVAL["Evaluation Harness<br/>10 standard + adversarial profiles<br/>precision@k, consistency,<br/>mood-ablation, RAG vs baseline"]
        GOLD[("Golden Set<br/>expected top songs per profile<br/>+ phrase to profile pairs<br/>for the parser")]
        REPORT["Eval Report<br/>pass/fail + metrics"]
        LOG[("Interaction Log<br/>ai_interactions.md")]
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
    QP --> RET
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

    %% ---------- STYLES ----------
    classDef existing fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef rag fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef rel fill:#ffedd5,stroke:#ea580c,color:#111827
    classDef human fill:#dcfce7,stroke:#16a34a,color:#111827
    classDef data fill:#f3f4f6,stroke:#9ca3af,color:#111827

    class LOAD,SCORE,RANK,EXPL,OUT,U existing
    class QP,RET,CAND rag
    class GUARD,UNIT,EVAL,REPORT rel
    class H1,H2,H3 human
    class CSV,KB,GOLD,LOG data
```

## How to read it

**Data flow (input, process, output)**
1. The user's request passes through the input guardrails, which clamp and normalize it.
2. The Claude-powered query parser turns the request into a `UserProfile`, checked against the known genre and mood labels. If there is no API key or the call fails, a keyword parser takes over. The local embedding retriever then uses the profile and the song index to pull candidate songs, matching on meaning rather than exact labels.
3. The existing `score_song()` and `recommend_songs()` rank the candidates. Scoring stays deterministic and explainable.
4. The output is the top-k list with a reason for each score.

**Testing**
- **Unit tests** (pytest) cover the deterministic parts: guardrails and scoring.
- The **evaluation harness** runs the 10 standard and adversarial profiles through the full pipeline. It compares results to a golden set and to the non-RAG baseline, then writes a report.

**Human checkpoints**
- **H1** checks that the parsed profile and retrieved candidates make sense before ranking, since this is where RAG can go wrong.
- **H2** is the user or developer reading the recommendations and explanations, with feedback logged.
- **H3** reviews the eval report and decides on weight changes or golden-set updates. Nothing changes automatically.
