"""Smart MCQ Solver — interactive demo of the Stage-A duplicate matcher.

Deploys the deterministic layer of the competition pipeline: TF-IDF near-
duplicate matching against the training set, resolving the answer by matching
answer TEXT (options are shuffled between duplicates, so matching the letter
would be wrong roughly four times in five).

The fine-tuned DeBERTa-v3-large and Qwen2.5-7B used for ranks 2-3 in the Kaggle
submission are not loaded here: together they need ~6 GB of weights and a GPU.
Stage A produces rank 1 on 479 of 500 test rows and scored 0.71446 standalone on
the leaderboard, so it is both the largest single contributor and the only part
that fits in a CPU container.

Vihaan Bhambhani (24f1002825) · BSDA2001P T2-2026
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LABELS = ["A", "B", "C", "D", "E"]
SIM_PIN = 0.93       # bar for a confident rank-1 pin
SIM_RELAXED = 0.70   # lower bar, reported as low confidence
TOPK = 25

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🎯",
                   layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp { max-width: 900px; margin: 0 auto; }
  .badge { display:inline-block; padding:3px 10px; border-radius:12px;
           font-size:0.78rem; font-weight:600; }
  .b-hi  { background:#e8f5e9; color:#1b5e20; }
  .b-mid { background:#fff4e1; color:#8a5300; }
  .b-lo  { background:#fdecea; color:#8a1c12; }
  .rank  { font-size:1.6rem; font-weight:700; color:#0b548c; }
</style>
""", unsafe_allow_html=True)


def norm(s) -> str:
    """Lowercase and collapse whitespace, for answer-text comparison."""
    return " ".join(str(s).lower().split())


def full_text(df: pd.DataFrame) -> pd.Series:
    """Prompt + all five options: the duplicate fingerprint."""
    return (df["prompt"].astype(str) + " "
            + df[LABELS].astype(str).agg(" ".join, axis=1))


@st.cache_resource(show_spinner="Building the TF-IDF index…")
def load_index():
    """Load train.csv once per container and fit the vectoriser."""
    for path in ("data/train.csv", "train.csv", "../data/train.csv"):
        if os.path.exists(path):
            train = pd.read_csv(path)
            break
    else:
        return None, None, None, None

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(full_text(train))
    answer_text = [norm(r[r["answer"]]) for _, r in train.iterrows()]
    return train, vec, X, answer_text


@st.cache_data
def prior_order(_train: pd.DataFrame) -> list[str]:
    """Answer letters ordered by training frequency (the fallback ranking)."""
    return _train["answer"].value_counts().index.tolist()


def predict(prompt, options, train, vec, X, answer_text):
    """Return (top3, method, confidence, evidence, votes)."""
    q = pd.DataFrame([{"prompt": prompt, **dict(zip(LABELS, options))}])
    sims = cosine_similarity(vec.transform(full_text(q)), X)[0]

    k = min(TOPK, len(sims))
    nbrs = np.argpartition(-sims, k - 1)[:k]
    nbrs = nbrs[np.argsort(-sims[nbrs])]
    best = float(sims[nbrs[0]])

    q_opts = [norm(o) for o in options]
    votes, evidence = {}, None
    bar = SIM_PIN if best >= SIM_PIN else SIM_RELAXED

    for j in nbrs:
        if sims[j] < bar:
            break
        ans = answer_text[j]
        if ans in q_opts:
            idx = q_opts.index(ans)
            votes[idx] = votes.get(idx, 0.0) + float(sims[j])
            if evidence is None:
                evidence = (train.iloc[j]["prompt"], train.iloc[j]["answer"],
                            train.iloc[j][train.iloc[j]["answer"]], float(sims[j]))

    order = [i for i, _ in sorted(votes.items(), key=lambda kv: -kv[1])]
    method = ("exact duplicate match" if best >= SIM_PIN and order
              else "weak partial match" if order else "class prior")

    for letter in prior_order(train):
        i = LABELS.index(letter)
        if i not in order:
            order.append(i)
    return [LABELS[i] for i in order[:3]], method, best, evidence, votes


# ── sidebar ────────────────────────────────────────────────────────────────
train, vec, X, answer_text = load_index()
if train is None:
    st.error("`data/train.csv` not found. Commit it so the app can build its index.")
    st.stop()

with st.sidebar:
    st.markdown("### How it works")
    st.markdown(f"""
1. Your question is fingerprinted as **prompt + all five options** and vectorised
   with TF-IDF over word 1- and 2-grams.
2. Cosine similarity is computed against all **{len(train):,}** training questions.
3. Neighbours above **{SIM_PIN}** vote for whichever of *your* options carries the
   same **answer text** as their correct answer.

Matching on *text* rather than *letter* is the crux: duplicate questions appear
with **shuffled options**, so copying the letter would be wrong about four times
in five.
""")
    st.markdown("---")
    st.markdown("### Metric")
    st.markdown("""
| True answer at | Score |
|---|---|
| Rank 1 | 1.000 |
| Rank 2 | 0.500 |
| Rank 3 | 0.333 |
| Not in top 3 | 0.000 |
""")
    st.caption("Random guessing = 0.3667 · Final pipeline = **0.76475**")
    st.markdown("---")
    st.caption("Vihaan Bhambhani · 24f1002825\nBSDA2001P · Term T2-2026")

# ── main ───────────────────────────────────────────────────────────────────
st.title("🎯 Smart MCQ Solver")
st.caption("Stage-A duplicate matcher · public leaderboard MAP@3 **0.76475**")

if "seed" not in st.session_state:
    st.session_state.seed = 0
    st.session_state.vals = ["", "", "", "", "", ""]

c1, c2 = st.columns([1, 3])
if c1.button("🎲 Load an example"):
    st.session_state.seed += 1
    row = train.sample(1, random_state=st.session_state.seed).iloc[0]
    st.session_state.vals = [row["prompt"]] + [row[c] for c in LABELS]
if c2.button("Clear"):
    st.session_state.vals = ["", "", "", "", "", ""]

prompt = st.text_area("Question", value=st.session_state.vals[0], height=90,
                      placeholder="Type or paste a multiple-choice question…")
cols = st.columns(2)
options = [cols[i % 2].text_input(f"Option {LABELS[i]}",
                                  value=st.session_state.vals[i + 1])
           for i in range(5)]

if st.button("Predict", type="primary", use_container_width=True):
    if not prompt.strip() or not all(o.strip() for o in options):
        st.warning("Please fill in the question and all five options.")
    else:
        top3, method, conf, evidence, votes = predict(
            prompt, options, train, vec, X, answer_text)

        badge = ("b-hi" if method == "exact duplicate match"
                 else "b-mid" if method == "weak partial match" else "b-lo")
        st.markdown(
            f"<span class='badge {badge}'>{method}</span> "
            f"&nbsp;&nbsp;best training-set similarity: <b>{conf:.4f}</b>",
            unsafe_allow_html=True)

        st.markdown("### Prediction")
        r1, r2, r3 = st.columns(3)
        for col, letter, rank, worth in zip((r1, r2, r3), top3, (1, 2, 3),
                                            ("1.000", "0.500", "0.333")):
            col.markdown(f"<div class='rank'>{rank}. {letter}</div>",
                         unsafe_allow_html=True)
            col.caption(f"MAP@3 if correct: {worth}")
            col.write(options[LABELS.index(letter)])

        if votes:
            st.markdown("##### Vote weights from matching training questions")
            vd = pd.DataFrame({
                "Option": [LABELS[i] for i in votes],
                "Answer text": [options[i] for i in votes],
                "Similarity-weighted votes": [round(v, 4) for v in votes.values()],
            }).sort_values("Similarity-weighted votes", ascending=False)
            st.dataframe(vd, hide_index=True, use_container_width=True)

        if evidence:
            with st.expander("Matched training question"):
                st.caption(f"cosine similarity {evidence[3]:.4f}")
                st.write(evidence[0])
                st.success(f"Its correct answer — **{evidence[1]}**: {evidence[2]}")
                st.caption("Note the letter may differ from the prediction above: "
                           "options are shuffled between duplicates, so the match "
                           "is made on answer text and re-located in your options.")
        else:
            st.info("No training question cleared the similarity bar, so the "
                    "class-prior ranking was used. On the competition test set "
                    "this happened for 21 of 500 rows.")

st.divider()
st.caption("Full pipeline (DeBERTa-v3-large + Qwen2.5-7B for ranks 2–3) requires "
           "a GPU and is not deployed here. Source: github.com/24f1002825-VihaanB")
