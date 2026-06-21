# Smart MCQ Solver — Retrieval-Augmented Multiple-Choice Answer Ranking

**Course:** Introduction to DL and GenAI Project \
**Name:** Vihaan Bhambhani \
**Roll No:** 24f1002825 \
**Term:** T2-2026 

---

## Project Overview 

Build an AI system that, given a multiple-choice question with five options (A–E),
predicts the **top three most probable answers** in ranked order. Submissions are
scored with **Mean Average Precision @ 3 (MAP@3)** on the Kaggle leaderboard.

The planned approach combines three required model types with a retrieval layer:

- **Classical baseline** — TF-IDF + LightGBM (Milestone 2)
- **From-scratch neural net** — a TextCNN with attention (Milestone 3)
- **Pre-trained transformer, fine-tuned** — DeBERTa-v3-large multiple-choice (Milestone 5)
- **Model of choice** — Qwen2.5-7B-Instruct zero-shot letter-logit scoring
- **RAG** — dense + lexical retrieval over a STEM Wikipedia corpus for context

> This README and the folder skeleton are the Milestone 0 setup. Code, notebooks,
> and reports are added as the project progresses.

---

## Repository Structure

```
dl-genai-mcq-solver/
├── notebooks/        # Kaggle / Colab notebooks (EDA, baselines, final inference)
├── src/              # Reusable scripts: train.py, inference.py, utils.py
├── data/             # Small data artefacts only (large data stays on Kaggle)
├── reports/          # Milestone reports + final technical report (PDF)
├── models/           # Saved model weights / adapters (or links to Kaggle Hub)
├── requirements.txt  # Pinned dependencies for reproducibility
├── .gitignore
└── README.md
```

---


## Git Workflow

The course enforces a milestone-branch workflow. `main` always holds the latest
stable, working version. **No milestone work is committed directly to `main`**, and
milestone branches are **never deleted**, even after merging.

```bash
# start a milestone
git checkout main
git pull origin main
git checkout -b milestone-1

# work, then commit with a meaningful message
git add .
git commit -m "Milestone 1: EDA, class distribution, baseline submission"
git push origin milestone-1

# after the milestone is complete, merge into main (keep the branch)
git checkout main
git merge milestone-1
git push origin main
```

**Commit messages:** describe what changed, e.g.
`Milestone 2: TF-IDF + LightGBM baseline, 5-fold CV` — not `update` / `final`.

---

## Reproducibility

```bash
pip install -r requirements.txt
```

All model runs are tracked in Weights & Biases; at least three runs are compared on
common metrics. Large datasets and weights live on Kaggle and are referenced, not
committed.
