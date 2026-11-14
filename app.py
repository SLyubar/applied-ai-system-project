"""Streamlit interface for the music recommender.

    streamlit run app.py

Type a request, read the recommendations and the confidence assessment, then say which
songs were right. Saving records the interaction and your rating in the interaction log
(logs/, git-ignored) and writes your song judgments into data/golden_set.json as a ranking
case. The command line (python -m src.main, src.evaluation, ...) works exactly as before.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)  # the project uses paths relative to the repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st  # noqa: E402

from src.evaluation import GOLDEN_PATH  # noqa: E402
from src.feedback import LOG_PATH, log_interaction, record_feedback  # noqa: E402
from src.golden_writer import find_ui_case, save_case  # noqa: E402
from src.pipeline import PARSERS, RETRIEVERS, run_query  # noqa: E402
from src.recommender import load_songs  # noqa: E402

# Overridable so tests (or a scratch copy of the golden set) never touch the real files.
LOG_FILE = os.environ.get("RECOMMENDER_LOG_FILE", LOG_PATH)
GOLDEN_FILE = os.environ.get("RECOMMENDER_GOLDEN_FILE", GOLDEN_PATH)

JUDGMENTS = ["No opinion", "Ideal", "Fine", "Must not appear"]
EXAMPLES = ["something mellow and acoustic to study to", "heavy music for when I am furious", "upbeat pop for a workout"]


@st.cache_data
def get_songs():
    return load_songs("data/songs.csv")


def search(query: str, parser: str, retriever: str, pool: int, k: int) -> None:
    result = run_query(query, get_songs(), parser=parser, retriever=retriever, pool=pool, k=k)
    result["interaction_id"] = log_interaction(
        result["query"], result["parser"], result["profile"], result["notes"],
        [{"id": r["id"], "title": r["title"], "score": r["score"]} for r in result["recommendations"]],
        LOG_FILE, result["retriever"], result["confidence"].as_dict(),
    )
    st.session_state["result"] = result


def show_result(result: dict) -> None:
    profile, conf = result["profile"], result["confidence"]
    acoustic = {True: "likes acoustic", False: "prefers non-acoustic"}.get(profile.get("likes_acoustic"), "no acoustic preference")
    st.caption(
        f"Understood as: **{profile['genre'] or 'any genre'}** · **{profile['mood'] or 'any mood'}** · "
        f"energy **{profile['energy']:.2f}** · {acoustic}  (parser: {result['parser']}, retriever: {result['retriever']})"
    )
    icon = {"high": "✅", "medium": "🟡", "low": "⚠️"}[conf.label]
    st.markdown(
        f"{icon} **Confidence: {conf.label}** ({conf.score:.2f}) — understanding {conf.understanding:.2f}, "
        f"match {conf.match:.2f}. Matched on: {', '.join(conf.criteria) or 'nothing'}."
    )
    for line in conf.reasons + result["notes"]:
        st.caption(f"• {line}")
    if conf.label == "low":
        st.warning("Low confidence: treat these as rough suggestions, not a good match.")


def feedback_form(result: dict) -> None:
    songs = get_songs()
    previous = find_ui_case(result["query"], GOLDEN_FILE)
    prior = {
        **{i: "Ideal" for i in previous.get("ideal_ids", [])},
        **{i: "Fine" for i in previous.get("acceptable_ids", [])},
        **{i: "Must not appear" for i in previous.get("forbidden_ids", [])},
    }
    if previous:
        st.info("You already judged this query; the form is pre-filled and saving will update that case.")

    with st.form(f"feedback-{result['interaction_id']}"):
        st.subheader("Your feedback")
        st.caption("Judge each song. Songs you leave at “No opinion” are not recorded.")
        judgments = {}
        for rank, rec in enumerate(result["recommendations"], start=1):
            with st.container(border=True):
                st.markdown(f"**{rank}. {rec['title']}** — {rec['artist']} · score {rec['score']:.2f}")
                st.caption("; ".join(rec["reasons"]))
                judgments[rec["id"]] = st.radio(
                    "Judgment", JUDGMENTS, index=JUDGMENTS.index(prior.get(rec["id"], "No opinion")),
                    horizontal=True, key=f"j-{result['interaction_id']}-{rec['id']}", label_visibility="collapsed",
                )
        shown = {r["id"] for r in result["recommendations"]}
        by_label = {f"{s['title']} — {s['artist']} (#{s['id']})": s["id"] for s in songs if s["id"] not in shown}
        by_id = {v: k for k, v in by_label.items()}
        missing = st.multiselect(
            "Songs that should have been recommended (optional)", list(by_label),
            default=[by_id[i] for i in previous.get("ideal_ids", []) if i in by_id],
        )
        overall = st.radio("Overall, were these good recommendations?", ["Skip", "Good", "Bad"], horizontal=True)
        note = st.text_input("Note (optional)", value=previous.get("notes", "") if previous else "")
        submitted = st.form_submit_button("Save feedback", type="primary")

    if not submitted:
        return
    messages = []
    if overall != "Skip":
        record_feedback(result["interaction_id"], overall.lower(), note, LOG_FILE)
        messages.append(f"Rated **{overall.lower()}** in the interaction log.")

    ideal = [i for i, j in judgments.items() if j == "Ideal"] + [by_label[m] for m in missing]
    fine = [i for i, j in judgments.items() if j == "Fine"]
    forbidden = [i for i, j in judgments.items() if j == "Must not appear"]
    if not (ideal or fine or forbidden):
        st.success(" ".join(messages) or "Nothing to save.")
        return
    try:
        case, action = save_case(result["query"], ideal, fine, forbidden, songs, note, GOLDEN_FILE)
    except ValueError as error:
        for m in messages:
            st.success(m)
        st.error(f"Golden set not changed: {error}")
        return
    messages.append(f"Golden set: **{action}** case “{case['name']}” "
                    f"({len(case['ideal_ids'])} ideal, {len(case['acceptable_ids'])} fine, {len(case['forbidden_ids'])} forbidden).")
    st.success(" ".join(messages))


def main() -> None:
    st.set_page_config(page_title="Music Recommender", page_icon="🎵")
    st.title("🎵 Music Recommender")

    with st.sidebar:
        st.header("Settings")
        parser = st.selectbox("Query parser", PARSERS, help="'gemini' calls the live API and falls back to keywords on failure.")
        retriever = st.selectbox("Retriever", RETRIEVERS, help="Narrow the catalog by description before scoring. 'gemini' uses live embeddings.")
        pool = st.slider("Retrieval pool", 5, 40, 20, disabled=retriever == "none")
        k = st.slider("Recommendations", 3, 10, 5)
        if "gemini" in (parser, retriever):
            st.caption("Gemini options use your API quota.")

    query = st.text_input("What are you in the mood for?", key="query", placeholder=EXAMPLES[0])
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        col.button(example, use_container_width=True, on_click=st.session_state.update, kwargs={"query": example})

    if st.button("Recommend", type="primary"):
        try:
            search(query, parser, retriever, pool, k)
        except ValueError:
            st.session_state.pop("result", None)
            st.error("Type a request first.")

    result = st.session_state.get("result")
    if result:
        st.divider()
        st.subheader(f"Results for “{result['query']}”")
        show_result(result)
        feedback_form(result)


main()
