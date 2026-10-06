# Smart MCQ Solver — Live Demo

Interactive deployment of the **Stage-A duplicate matcher** from my
Introduction to Deep Learning & GenAI project (BSDA2001P, T2-2026).

**Live app:** `https://dl-genai-project-deployment-24f1002825.streamlit.app`
**Full project:** https://github.com/24f1002825-VihaanB/DL-GenAI-Project-Deployment

---

## What this demonstrates

The competition dataset has a decisive property: **98.2% of test questions have a
near-duplicate in the training set** at cosine similarity ≥ 0.93, and duplicates
appear with **shuffled option order**. Stage A exploits this by matching each
question to its training twin and resolving the answer by **answer text rather
than answer letter** — copying the letter would be wrong roughly four times in
five.

This layer alone pins rank 1 on **479 of 500** test rows and scored **0.71446**
standalone on the public leaderboard. The complete pipeline scored **0.76475**.

## Why only Stage A is deployed

Ranks 2–3 in the full pipeline come from a fine-tuned DeBERTa-v3-large blended
with a 4-bit Qwen2.5-7B-Instruct — about 6 GB of weights, requiring a GPU. No
free hosting tier provides that. Stage A is pure CPU and needs only a 2 MB CSV,
so it is the part that can be served honestly and instantly.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application |
| `requirements.txt` | Four packages, no torch — installs in ~60 s |
| `data/train.csv` | Knowledge base for matching (~2 MB) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), grant access to the repo
   (Settings → Linked accounts → Manage permissions) if it is private.
3. New app → select the repo, branch `main`, main file `app.py`.
4. Deploy. Verify the URL loads in an incognito window.

## Metric

MAP@3 — the true answer scores 1.000 at rank 1, 0.500 at rank 2, 0.333 at rank 3,
and 0 otherwise. Uniform random guessing yields 0.3667.
