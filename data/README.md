# data/

Competition data is **not committed** — it is distributed by Kaggle and excluded
by `.gitignore` (`data/*.csv`). This file documents what belongs here and where
to get it.

## Getting the data

**On Kaggle**, the competition dataset mounts automatically:

```
/kaggle/input/competitions/smart-mcq-solver-challenge/
├── train.csv
└── test.csv
```

**Locally**, download both files from the competition's Data tab and place them
in this folder:

```
data/
├── train.csv
└── test.csv
```

`src/utils.resolve_data_dir()` checks the Kaggle mount first, then the legacy
`/kaggle/input/<slug>` layout, then this folder — so the same code runs in both
environments without edits.

## Schema

| Column | Type | Description |
|---|---|---|
| `id` | int | Row identifier |
| `prompt` | str | The question stem |
| `A` – `E` | str | The five answer options (full text, not just labels) |
| `answer` | str | Correct option letter — **train only** |

| Split | Rows |
|---|---|
| `train.csv` | 2,000 |
| `test.csv` | 500 |

Answer distribution in train is mildly skewed: B 24.5%, C 23.0%, A 18.5%,
D 17.9%, E 16.2%. A static `B C A` guess therefore scores MAP@3 = 0.4213,
above the 0.3667 random baseline.

## The property this project depends on

**98.2% of test questions have a near-duplicate in train** at cosine ≥ 0.93, and
383 of 500 exceed 0.99. Duplicates appear with **shuffled option order**, so the
correct answer must be matched by its *text* rather than its letter — copying the
letter would be wrong roughly four times in five.

This is a property of the provided competition data. No external data and no test
labels are used anywhere in the pipeline. `src/preprocess.py` exploits it
directly, and the analysis is in Section 3 of the competition notebook.

## Files written here

None. All artefacts are written to `models/`.
