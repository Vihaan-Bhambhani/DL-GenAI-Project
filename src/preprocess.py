"""Stage A: TF-IDF near-duplicate matching.

The competition's test set overlaps heavily with train: 98.2% of test rows have
a near-duplicate in train at cosine >= 0.93. Stage A exploits that directly by
matching each test row to its train twin and reading off the twin's answer
*text* (not letter, since option order is shuffled between duplicates).

Usage
-----
    python -m src.preprocess --out artifacts/
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import LABELS, load_data, set_seed

SIM_PIN = 0.93          # cosine bar for trusting a match at rank 1
                        # (matches the value used in the final notebook)
TOPK_NEIGHBOURS = 25    # train neighbours inspected per query row


def norm(s) -> str:
    """Lowercase and collapse whitespace, for answer-text comparison."""
    return " ".join(str(s).lower().split())


def full_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate prompt and all five options into one fingerprint string."""
    return (df["prompt"].astype(str) + " "
            + df[LABELS].astype(str).agg(" ".join, axis=1))


def rank_candidates(query_df, index_df, S, sim_pin: float = SIM_PIN,
                    topk: int = TOPK_NEIGHBOURS):
    """Vote for each query row's answer using its nearest rows in ``index_df``.

    Every neighbour above ``sim_pin`` votes for whichever query option carries
    the same text as that neighbour's correct answer, weighted by similarity.
    Matching on answer TEXT is what makes this robust to shuffled options.

    Returns
    -------
    list[list[tuple[int, float]]]
        Per row, (option_index, vote_weight) sorted best first; empty if the
        row has no match above the bar.
    """
    query_opts = [[norm(r[c]) for c in LABELS]
                  for _, r in query_df[LABELS].iterrows()]
    index_ans = [norm(r[r["answer"]]) for _, r in index_df.iterrows()]

    out = []
    for i in range(len(query_df)):
        sims = S[i]
        k = min(topk, len(sims))
        nbrs = np.argpartition(-sims, k - 1)[:k]
        nbrs = nbrs[np.argsort(-sims[nbrs])]

        q_opts = query_opts[i]
        votes: dict[int, float] = {}
        for j in nbrs:
            if sims[j] < sim_pin:
                break
            ans = index_ans[j]
            if ans in q_opts:
                idx = q_opts.index(ans)
                votes[idx] = votes.get(idx, 0.0) + float(sims[j])
        out.append(sorted(votes.items(), key=lambda kv: -kv[1]))
    return out


def build_duplicate_pins(train: pd.DataFrame, test: pd.DataFrame):
    """Return (pinned, max_sim) arrays for the test set.

    ``pinned[i]`` is the option index chosen for test row i, or -1 when no
    train match clears the similarity bar.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(pd.concat([full_text(train), full_text(test)]))
    Xtr, Xte = vec.transform(full_text(train)), vec.transform(full_text(test))

    S = cosine_similarity(Xte, Xtr)
    cands = rank_candidates(test, train, S)
    pinned = np.array([c[0][0] if c else -1 for c in cands], dtype=int)
    max_sim = S.max(axis=1)
    return pinned, max_sim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts",
                    help="directory for the .npy artefacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    set_seed()
    train, test = load_data()
    pinned, max_sim = build_duplicate_pins(train, test)

    np.save(os.path.join(args.out, "dup_pinned.npy"), pinned)
    np.save(os.path.join(args.out, "dup_max_sim.npy"), max_sim)
    n = int((pinned >= 0).sum())
    print(f"pinned {n}/{len(test)} test rows ({n / len(test):.1%})")


if __name__ == "__main__":
    main()
