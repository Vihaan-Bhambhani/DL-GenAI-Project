# 🎯 Smart MCQ Solver

### Hybrid NLP, Retrieval & LLM System for Multiple-Choice Answer Ranking

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://smart-mcq-solver-vihaan.streamlit.app/)
[![Source Code](https://img.shields.io/badge/Source-GitHub-181717?logo=github&logoColor=white)](https://github.com/Vihaan-Bhambhani/DL-GenAI-Project)

**Live Demo:** https://smart-mcq-solver-vihaan.streamlit.app/

**Source Code:** https://github.com/Vihaan-Bhambhani/DL-GenAI-Project

Smart MCQ Solver is an end-to-end machine learning system for **top-3 answer ranking in multiple-choice questions**.

The project combines:

- Information retrieval
- TF-IDF similarity search
- Feature-based machine learning
- Neural text modeling
- Transformer fine-tuning
- Large language models
- Ensemble ranking
- Streamlit deployment

The final pipeline achieved a **0.76475 public leaderboard MAP@3**.

---

## 🚀 Project Highlights

| Metric | Result |
|---|---:|
| Public leaderboard MAP@3 | **0.76475** |
| Training questions | **2,000** |
| Test questions | **500** |
| Near-duplicate test questions at cosine ≥ 0.93 | **98.2%** |
| Test rows pinned to rank 1 by Stage A | **479 / 500** |
| Random MAP@3 baseline | **0.3667** |
| Majority-class baseline | **0.4213** |

---

# 🔎 The Key Insight

The most important breakthrough in the project came from **understanding the structure of the data before selecting the final model architecture**.

Exploratory analysis revealed that **98.2% of the test questions had a near-duplicate in the training set** at cosine similarity ≥ 0.93, with 383 of 500 questions exceeding 0.99 similarity.

However, duplicate questions frequently appeared with their answer choices in a different order.

For example:

```text
Training question

A → Paris
B → London
C → Berlin
D → Rome
E → Madrid


Test duplicate

A → London
B → Rome
C → Madrid
D → Paris
E → Berlin
```

Simply copying the training answer letter would therefore produce the wrong answer.

Instead, the retrieval layer:

1. Finds the closest training question.
2. Retrieves its known correct answer as **text**.
3. Searches the current question's five options for the same answer text.
4. Maps that answer back to the correct option letter for the current question.

This observation became the foundation of the retrieval layer and strongly influenced the final architecture.

---

# 🧠 System Architecture

```text
                         Question + 5 Options
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ Text Representation     │
                    │ TF-IDF word 1/2-grams   │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 │                                │
                 ▼                                ▼
       ┌─────────────────────┐          ┌─────────────────────┐
       │ Stage A             │          │ Learned Models      │
       │ Duplicate Retrieval │          │                     │
       │                     │          │ LightGBM            │
       │ Cosine Similarity   │          │ TextCNN + Attention │
       │ Answer-text Match   │          │ DeBERTa-v3-large    │
       │ Similarity Voting   │          │ Qwen2.5-7B          │
       └──────────┬──────────┘          └──────────┬──────────┘
                  │                                │
                  └───────────────┬────────────────┘
                                  ▼
                         ┌──────────────────┐
                         │ Final Top-3 Rank │
                         └──────────────────┘
```

The final submission used the duplicate matcher to establish the strongest rank-1 prediction and a **DeBERTa + Qwen blend** to provide additional ranking information for positions 2 and 3.

---

# 🔬 Retrieval Layer

The duplicate matcher uses:

- TF-IDF word 1- and 2-grams
- Cosine similarity
- Top-25 nearest training questions
- Similarity thresholds of 0.93 and 0.70
- Answer-text matching instead of answer-letter copying
- Similarity-weighted voting
- Training-label frequency as a fallback prior

The retrieval layer is deterministic and lightweight enough to run entirely on CPU infrastructure.

It is also the component powering the public Streamlit deployment.

### Why answer-text matching matters

The system does **not** assume that the correct answer will remain in the same option position between duplicate questions.

Instead:

```text
Retrieved training answer
        ↓
Answer text
        ↓
Search current question's options
        ↓
Find matching option
        ↓
Return current option letter
```

This makes the retrieval step robust to shuffled answer choices.

---

# 🤖 Machine Learning Models

## LightGBM

A feature-based model trained using **55 engineered similarity and text features** derived from the question, answer options, and retrieved training examples.

## TextCNN + Attention

A from-scratch neural architecture using convolutional representations and attention pooling for multiple-choice text classification.

## DeBERTa-v3-large

A pretrained transformer fine-tuned for semantic answer prediction using **5-fold cross-validation**.

## Qwen2.5-7B-Instruct

A quantized large language model used as a second semantic model and blended with DeBERTa for the final ranking stage.

---

# 🧪 RAG Experiments

Retrieval-Augmented Generation was evaluated as part of the experimentation process.

| Retrieval configuration | OOF MAP@3 | Public LB |
|---|---:|---:|
| Closed-book | **0.8888** | **0.76475** |
| RAG — merged query | 0.7678 | 0.74023 |
| RAG — per-option query | 0.8370 | 0.74688 |

The experiments showed that adding external retrieved context did not reliably improve performance on this dataset.

The final system therefore **retains retrieval for duplicate detection while keeping external RAG context disabled during transformer inference**.

This was an important engineering decision: the architecture was determined through measured experiments and ablation results rather than assuming that adding RAG would automatically improve performance.

---

# 📊 Results

Each major component was evaluated through separate competition submissions.

| Component | Public MAP@3 |
|---|---:|
| DeBERTa-v3-large | 0.62136 |
| Duplicate matcher | 0.71446 |
| DeBERTa + Qwen blend | 0.72069 |
| LightGBM | 0.74397 |
| TextCNN + Attention | 0.75353 |
| **Final pipeline** | **0.76475** |

### Validation caveat

LightGBM and TextCNN produced unusually high out-of-fold scores.

Further investigation showed that duplicate clusters could span validation folds, allowing models to effectively memorize relationships that also existed in the explicit retrieval structure.

The competition leaderboard was therefore treated as the more meaningful external evaluation rather than presenting inflated validation scores as genuine generalization.

---

# 🛠️ Engineering Challenges & Solutions

Several practical engineering issues were identified and resolved during development.

### Mixed-precision training

A dependency change affected transformer loading behaviour and caused:

```text
Attempting to unscale FP16 gradients
```

The training configuration was adjusted so model-loading precision and mixed-precision computation were handled separately.

### Multi-GPU memory pressure

Automatic multi-GPU replication caused excessive memory usage on the primary device.

The training setup was adjusted to reduce unnecessary replication and enable gradient checkpointing.

### Retrieval dataset compatibility

Changes in Kaggle dataset mounting required support for multiple dataset layouts and Hugging Face Arrow-based datasets.

### Variable shadowing

A retrieval fallback used `nn` as a variable name, conflicting with `torch.nn`.

The index variable was renamed to eliminate the namespace collision.

---

# 🌐 Live Deployment

The lightweight retrieval layer is publicly deployed using Streamlit.

## Live Demo

### https://smart-mcq-solver-vihaan.streamlit.app/

The application provides:

- Example question loading
- Multiple-choice input
- Top-3 answer prediction
- Best similarity score
- Similarity-weighted vote information
- Matched training-question evidence
- Explanation of the retrieval decision

### Demo workflow

```text
Load an example
       ↓
Question + five options
       ↓
TF-IDF vectorisation
       ↓
Similarity search
       ↓
Near-duplicate retrieval
       ↓
Answer-text matching
       ↓
Top-3 prediction
       ↓
Evidence + vote weights
```

The public deployment focuses on the lightweight Stage-A retrieval component because it can run on CPU-compatible infrastructure.

The complete competition pipeline contains transformer and LLM components that require substantially more GPU and memory resources.

---

# 📁 Repository Structure

```text
DL-GenAI-Project/
│
├── app/
│   └── streamlit/
│       ├── app.py
│       ├── data/
│       │   └── train.csv
│       ├── requirements.txt
│       └── runtime.txt
│
├── data/
│   ├── README.md
│   └── .gitkeep
│
├── models/
│   ├── README.md
│   └── .gitkeep
│
├── notebooks/
│   ├── milestone-1.ipynb
│   ├── milestone-2.ipynb
│   ├── milestone-3.ipynb
│   ├── milestone-4.ipynb
│   ├── milestone-5.ipynb
│   ├── experiments/
│   │   └── dl-24f1002825-notebook-t22026_RAG.ipynb
│   └── dl-24f1002825-notebook-t22026.ipynb
│
├── reports/
│   └── DL_GenAI_Project_Report_24f1002825.pdf
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ▶️ Run the Demo Locally

The Streamlit application uses a lightweight dependency environment.

```bash
git clone https://github.com/Vihaan-Bhambhani/DL-GenAI-Project.git
cd DL-GenAI-Project

python -m pip install -r app/streamlit/requirements.txt

python -m streamlit run app/streamlit/app.py
```

The application expects its demo knowledge base at:

```text
app/streamlit/data/train.csv
```

---

# ⚙️ Reproduce the Full Pipeline

The complete experimentation and training environment is defined in:

```text
requirements.txt
```

The full pipeline includes GPU-dependent transformer training.

Example workflow:

```bash
python -m src.preprocess --out models/
python -m src.train --folds 5 --epochs 2 --out models/
python -m src.inference --artifacts models/ --out submission.csv
```

The notebooks contain the complete experimentation workflow, milestone progression, model experiments, evaluation, and final submission development.

---

# 📜 License

The repository's source code is released under the **MIT License**.

The competition dataset bundled with the public demo at
`app/streamlit/data/train.csv` is **not covered by the code license** and remains
subject to the terms under which that dataset was provided.

---

# 📌 Limitations

The strongest retrieval-based advantage comes from the unusually high degree of train/test duplication present in this particular dataset.

A dataset without those duplicate or near-duplicate relationships would place substantially more weight on the semantic models.

The deployed application therefore demonstrates the lightweight retrieval component rather than presenting it as a replacement for the complete GPU-based pipeline.

---

# 🎯 What This Project Demonstrates

This project covers an end-to-end applied machine learning workflow:

**Problem discovery → EDA → hypothesis formation → retrieval design → feature engineering → model experimentation → ablation studies → ensemble design → failure analysis → deployment**

The main engineering lesson was:

> **Better modeling starts with understanding the structure of the data.**

Rather than immediately optimizing for a larger model, the project first identified a structural property of the dataset and designed the system around it.

---

## Project Origin

Originally developed for the Kaggle **Smart MCQ Solver Challenge** as part of an applied Deep Learning & GenAI project.

**Author:** Vihaan Bhambhani
