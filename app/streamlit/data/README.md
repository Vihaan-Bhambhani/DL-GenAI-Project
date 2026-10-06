# data/

Place the competition training file here as `train.csv`.

It must contain the columns `id, prompt, A, B, C, D, E, answer` — the app builds
its TF-IDF index from `prompt` plus the five option columns, and reads the
correct answer's *text* (not its letter) for matching.

Approximately 2 MB, so it is committed rather than gitignored: the app cannot
start without it.
