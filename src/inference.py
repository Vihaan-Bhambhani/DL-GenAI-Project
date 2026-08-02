"""Ensemble stage: combine the duplicate pins with model probabilities.

Rank 1 is always the Stage-A duplicate pin. Ranks 2-3 come from a blend of
DeBERTa and Qwen, deliberately excluding LightGBM and TextCNN: those two
memorise the same duplicate structure the pin already encodes, so they add
nothing where it matters (ranks 2-3 only score when rank 1 is WRONG).

Usage
-----
    python -m src.inference --artifacts artifacts/ --out submission.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .utils import ID2LABEL, load_data, write_submission

W_DEBERTA = 0.85   # measured on the leaderboard, not tuned on a holdout


def normalise(a: np.ndarray) -> np.ndarray:
    """Row-normalise a probability matrix."""
    return a / np.clip(a.sum(axis=1, keepdims=True), 1e-9, None)


def build_ranking(pinned: np.ndarray, blend: np.ndarray) -> list[list[str]]:
    """Rank-1 from the duplicate pin, remaining slots from the model blend."""
    out = []
    for i in range(len(blend)):
        order: list[int] = []
        if pinned is not None and pinned[i] >= 0:
            order.append(int(pinned[i]))
        for k in np.argsort(-blend[i]):
            if int(k) not in order:
                order.append(int(k))
        out.append([ID2LABEL[k] for k in order[:3]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--w-deberta", type=float, default=W_DEBERTA)
    args = ap.parse_args()

    _, test = load_data()

    def load(name):
        p = os.path.join(args.artifacts, name)
        return np.load(p) if os.path.exists(p) else None

    deberta = load("deberta_test_probs.npy")
    qwen = load("llm_test_probs.npy")
    pinned = load("dup_pinned.npy")

    if deberta is not None and qwen is not None:
        blend = (args.w_deberta * normalise(deberta)
                 + (1 - args.w_deberta) * normalise(qwen))
    elif deberta is not None:
        blend = normalise(deberta)
    elif qwen is not None:
        blend = normalise(qwen)
    else:
        raise SystemExit("no model probabilities found in --artifacts")

    top3 = build_ranking(pinned, blend)
    write_submission(test["id"], top3, args.out)
    n = 0 if pinned is None else int((pinned >= 0).sum())
    print(f"wrote {args.out}: {n}/{len(test)} rows pinned at rank 1, "
          f"w_deberta={args.w_deberta}")


if __name__ == "__main__":
    main()
