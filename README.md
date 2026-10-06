# Smart MCQ Solver — Retrieval-Augmented Multiple-Choice Question Answering

**Vihaan Bhambhani · 24f1002825**
Introduction to Deep Learning & GenAI (BSDA2001P) · Term T2-2026

Kaggle: *Smart MCQ Solver Challenge* — given a question prompt and five options
(A–E), predict the three most likely answers. Scored on **MAP@3**.

**Public leaderboard score: 0.76475** (cutoff to qualify: 0.73)

---

## Problem

| | |
|---|---|
| Train / test | 2,000 / 500 rows |
| Options | 5 per question (A–E) |
| Metric | MAP@3 — 1.0 at rank 1, 0.5 at rank 2, 0.333 at rank 3, else 0 |
| Random baseline | 0.3667 |
| Majority-class baseline | 0.4213 |

---

## Key finding

EDA revealed the property that drives the whole solution: **98.2% of test
questions have a near-duplicate in the training set** (cosine ≥ 0.93), and 383
of 500 exceed 0.99. Crucially, duplicates appear with **shuffled options**, so
the answer must be matched by *text*, not by letter.

This is a property of the provided competition data — no external data and no
test labels are used.

---

## Architecture

```
train.csv / test.csv
        │
        ├── Stage A: TF-IDF duplicate matcher ──► RANK 1  (479/500 rows pinned)
        │
        ├── Stage B: RAG retrieval (MiniLM + FAISS + TF-IDF rerank)
        │            measured, disabled — see ablation below
        │
        ├── Model 1: LightGBM, 55 engineered similarity features
        ├── Model 2: TextCNN + attention pooling      (from-scratch NN)
        ├── Model 3a: DeBERTa-v3-large, 5-fold CV     (pretrained)
        └── Model 3b: Qwen2.5-7B-Instruct, 4-bit      (model of choice)
                    │
                    └── DeBERTa/Qwen blend (w=0.85) ──► RANKS 2–3
                                    │
                                submission.csv
```

**Why only two models feed ranks 2–3.** Ranks 2 and 3 only earn points on rows
where rank 1 is *wrong*, so what matters there is being **independent** of the
duplicate matcher, not being accurate. LightGBM and TextCNN both memorised the
same duplicate structure — proven because pinning rank 1 on top of TextCNN
changed its leaderboard score by exactly zero (0.74646 either way). Only DeBERTa
and Qwen disagree with the pin often enough to add value.

---

## Results

### Component ablations (each a separate Kaggle submission)

| Component | Public LB |
|---|---|
| DeBERTa alone | 0.62136 |
| Duplicate matcher alone | 0.71446 |
| DeBERTa + Qwen blend | 0.72069 |
| LightGBM alone | 0.74397 |
| TextCNN alone | 0.75353 |
| **Full pipeline** | **0.76475** |

### Model metrics

| Model | OOF MAP@3 | Accuracy | Macro-F1 |
|---|---|---|---|
| LightGBM | 0.9998 | 0.9995 | 0.9995 |
| TextCNN + Attention | 0.9934 | 0.9870 | 0.9870 |
| DeBERTa-v3-large | 0.8888 | 0.8110 | 0.8096 |
| Qwen2.5-7B (holdout) | 0.8950 | — | — |

> LightGBM's and TextCNN's near-perfect OOF scores are **not** genuine skill.
> Duplicate clusters span CV folds, so both models memorise the same lookup the
> matcher performs explicitly. Their true leaderboard scores are 0.74397 and
> 0.75353. This is why blend weights were set from leaderboard measurements
> rather than tuned on OOF.

### RAG ablation (DeBERTa OOF, all at 512-token sequences)

| Retrieval configuration | OOF MAP@3 | Public LB |
|---|---|---|
| Closed-book (no retrieval) | **0.8888** | **0.76475** |
| RAG, merged query (prompt + all 5 options) | 0.7678 | 0.74023 |
| RAG, per-option query | 0.8370 | 0.74688 |

Per-option retrieval recovered +0.069 OOF over the merged query, confirming
retrieval *quality* was the issue — but closed-book still wins. These are
generated MCQs whose answers are not reliably present in Wikipedia, so retrieved
passages act as distractors. Retrieval is implemented and runs; feeding it to
the models is disabled via `USE_RAG_CONTEXT = False`.

---

## Repository structure

```
├── notebooks/
│   ├── milestone-1.ipynb … milestone-5.ipynb
│   └── dl-24f1002825-notebook-t22026.ipynb   ← final competition notebook
├── src/
│   ├── utils.py        metric, constants, submission writer
│   ├── preprocess.py   Stage A duplicate matcher
│   ├── train.py        DeBERTa K-fold training
│   └── inference.py    ensembling and submission
├── data/               (gitignored — see data/README.md)
├── models/             (gitignored — see models/README.md)
├── reports/            final technical report
└── requirements.txt    pinned dependencies
```

---

## Setup

```bash
git clone https://github.com/Vihaan-Bhambhani/DL-GenAI-Project.git
cd DL-GenAI-Project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `train.csv` and `test.csv` in `data/` (see `data/README.md`).

## Reproducing the submission

```bash
python -m src.preprocess --out models/                  # Stage A → dup_pinned.npy
python -m src.train --folds 5 --epochs 2 --out models/  # DeBERTa CV (GPU)
python -m src.inference --artifacts models/ --out submission.csv
```

Or run `notebooks/dl-24f1002825-notebook-t22026.ipynb` end-to-end on Kaggle
(GPU T4×2, Internet on, ~90 min).

**Dependencies are pinned deliberately.** A mid-project Kaggle upgrade to
transformers v5 changed the `from_pretrained` dtype default to `"auto"`, loading
fp16 weights and breaking mixed-precision training with *"Attempting to unscale
FP16 gradients"*. Pinning prevents a recurrence.

---

## Experiment tracking

All runs logged to W&B project **`24f1002825-t22026`** — 8 runs covering Stage A,
Stage B, all four models, and a cross-model comparison. Every model reports the
same three metric keys (`overall_map3`, `accuracy`, `f1_macro`) so runs are
directly comparable.

---

## Notable engineering issues resolved

| Issue | Cause | Fix |
|---|---|---|
| `Attempting to unscale FP16 gradients` | transformers v5 changed the dtype default to `"auto"` | pin `dtype=torch.float32`; fp16 for the forward pass only |
| CUDA OOM in `replica 0 on device 0` | Trainer auto-wraps in DataParallel on 2×T4; device 0 carries replicas *and* gathered outputs | `args._n_gpu = 1` + gradient checkpointing |
| RAG silently retrieving nothing | Kaggle moved dataset mounts to `/kaggle/input/datasets/<owner>/<slug>`; corpora ship as HF `.arrow`, not parquet | glob both layouts; `load_from_disk` for arrow |
| `'NearestNeighbors' object has no attribute 'Module'` | a FAISS fallback bound `nn = NearestNeighbors(...)`, shadowing `torch.nn` | rename to `nn_index` |

---

## Limitations

- **Local validation does not predict the leaderboard.** Splitting train in half
  still leaves each row's duplicate twin in the index half, so the matcher
  reports 100% precision while the leaderboard implies ~64%. A cluster-aware
  split — holding out entire duplicate clusters — would fix this.
- The approach depends on train/test duplication and would not transfer to a
  dataset without it.
- Deployment runs Stage A only; the transformer models need a GPU.
