# models/

Model weights and prediction arrays are **not committed** — checkpoints run to
hundreds of MB and are excluded by `.gitignore` (`models/*.bin`,
`models/*.safetensors`, `models/*.pt`, `*.npy`).

This file documents what the pipeline writes here, so the `.npy` filenames that
appear in `src/inference.py` can be traced to the script that produced them.

## Artefacts

| File | Written by | Shape | Contents |
|---|---|---|---|
| `dup_pinned.npy` | `src/preprocess.py` | (500,) | Option index pinned per test row; `-1` if no match cleared the similarity bar |
| `dup_max_sim.npy` | `src/preprocess.py` | (500,) | Best train-match cosine per test row |
| `deberta_oof_probs.npy` | `src/train.py` | (2000, 5) | Out-of-fold probabilities |
| `deberta_test_probs.npy` | `src/train.py` | (500, 5) | Test probabilities, averaged over folds |
| `llm_test_probs.npy` | Qwen stage (notebook §9) | (500, 5) | Zero-shot test probabilities |
| `llm_holdout_probs.npy` | Qwen stage (notebook §9) | (300, 5) | Holdout probabilities for weight tuning |

Column order in every probability array matches `LABELS = ["A","B","C","D","E"]`.

`src/inference.py` skips any artefact that is absent and falls back accordingly,
so a partial pipeline still produces a valid submission — but check the printed
summary, because a missing DeBERTa file silently degrades the blend to Qwen only.

## Pretrained weights used

None are stored in this repo; all are fetched at runtime.

| Model | Source | Size |
|---|---|---|
| DeBERTa-v3-large | `microsoft/deberta-v3-large` (HF Hub) | ~1.6 GB |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4 | ~4.5 GB quantised |
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB |
| GloVe 6B 100d | Kaggle: `rtatman/glove-global-vectors-for-word-representation` | ~350 MB |

On Kaggle, keep **Internet: On** so the Hub models download. GloVe is attached as
a dataset and resolved from `/kaggle/input/datasets/<owner>/<slug>`.

## Reproducing

```bash
python -m src.preprocess --out models/                  # CPU, ~30 s
python -m src.train --folds 5 --epochs 2 --out models/  # GPU, ~90 min
python -m src.inference --artifacts models/ --out submission.csv
```

The Qwen stage runs only in the competition notebook (§9) — it needs 4-bit
quantisation on a GPU and is not part of the `src/` entry points.
