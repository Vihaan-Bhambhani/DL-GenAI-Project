"""Shared constants, metrics and helpers for the Smart MCQ Solver pipeline.

Every stage of the pipeline imports from this module so that the metric,
the label ordering and the submission format are defined exactly once.

Author: Vihaan Bhambhani (24f1002825)
"""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ── Constants ───────────────────────────────────────────────────────────────
SEED = 42
LABELS = ["A", "B", "C", "D", "E"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Kaggle mounts competition data here; the second path is the legacy layout.
DATA_DIR_CANDIDATES = [
    "/kaggle/input/competitions/smart-mcq-solver-challenge",
    "/kaggle/input/smart-mcq-solver-challenge",
    "data",
]


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (including CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_data_dir() -> str:
    """Return the first existing competition-data directory.

    Raises
    ------
    FileNotFoundError
        If none of the candidate locations contain train.csv.
    """
    for d in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(d, "train.csv")):
            return d
    raise FileNotFoundError(
        f"train.csv not found in any of: {DATA_DIR_CANDIDATES}")


def load_data(data_dir: str | None = None):
    """Load the competition train and test splits as DataFrames."""
    data_dir = data_dir or resolve_data_dir()
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test


# ── Metrics ─────────────────────────────────────────────────────────────────
def map_at_3(y_true, y_pred_top3) -> float:
    """Mean Average Precision @ 3, the competition metric.

    If the true label appears at rank k (1-indexed) among the top 3
    predictions the sample scores 1/k; a miss scores 0.

    Parameters
    ----------
    y_true : array-like of str
        Ground-truth letters, e.g. ``['A', 'C', ...]``.
    y_pred_top3 : list[list[str]]
        Per-sample ranked predictions, best first.

    Returns
    -------
    float
        Mean score over all samples, in [0, 1].

    Examples
    --------
    >>> round(map_at_3(['A', 'B'], [['A', 'B', 'C'], ['X', 'B', 'C']]), 4)
    0.75
    """
    scores = []
    for true, preds in zip(y_true, y_pred_top3):
        score = 0.0
        for rank, pred in enumerate(preds[:3]):
            if pred == true:
                score = 1.0 / (rank + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def top3_from_probs(probs: np.ndarray) -> list[list[str]]:
    """Convert an (n, 5) probability matrix to ranked letter predictions."""
    return [[ID2LABEL[i] for i in np.argsort(-p)[:3]] for p in probs]


def evaluate_predictions(y_true, probs: np.ndarray) -> dict:
    """Score predictions with the three metrics tracked across all models.

    MAP@3 is the competition target; accuracy and macro-F1 are computed on the
    top-1 prediction so that every W&B run can be compared on common keys.
    """
    top3 = top3_from_probs(probs)
    top1 = [t[0] for t in top3]
    return {
        "map3": map_at_3(y_true, top3),
        "accuracy": accuracy_score(y_true, top1),
        "f1_macro": f1_score(y_true, top1, average="macro",
                             labels=LABELS, zero_division=0),
    }


# ── Submission ──────────────────────────────────────────────────────────────
def write_submission(ids, top3: list[list[str]], path: str) -> pd.DataFrame:
    """Write a competition submission and assert it is well formed.

    The format is ``ID,Prediction`` where Prediction is three distinct,
    space-separated letters ordered best-first.
    """
    preds = [" ".join(t[:3]) for t in top3]
    sub = pd.DataFrame({"ID": ids, "Prediction": preds})

    assert all(len(p.split()) == 3 for p in sub["Prediction"]), \
        "every row must contain exactly 3 predictions"
    assert all(all(x in LABELS for x in p.split()) for p in sub["Prediction"]), \
        "predictions must be drawn from A-E"
    assert all(len(set(p.split())) == 3 for p in sub["Prediction"]), \
        "predictions within a row must be distinct"

    sub.to_csv(path, index=False)
    return sub


def safe_wandb_init(**kwargs):
    """Initialise a W&B run, falling back to offline mode on failure."""
    import wandb

    try:
        return wandb.init(**kwargs)
    except Exception as exc:  # noqa: BLE001 - offline fallback is intentional
        print(f"W&B unavailable ({exc}); continuing offline.")
        os.environ["WANDB_MODE"] = "offline"
        try:
            return wandb.init(**kwargs)
        except Exception:
            return None
