"""Training entry point for the DeBERTa-v3-large multiple-choice model.

Trains a StratifiedKFold ensemble and writes out-of-fold and test
probabilities for the ensembling stage.

Usage
-----
    python -m src.train --folds 5 --epochs 2 --out artifacts/

Notes
-----
* ``dtype=torch.float32`` is pinned explicitly. transformers v5 changed the
  ``from_pretrained`` default to ``dtype="auto"``, which loads deberta-v3-large
  in fp16 and makes GradScaler raise "Attempting to unscale FP16 gradients".
  Mixed precision needs fp32 master weights with fp16 only in the forward pass.
* ``args._n_gpu = 1`` disables DataParallel. Kaggle exposes 2x T4 and Trainer
  wraps the model automatically, but DataParallel replicates onto cuda:0 and
  gathers outputs there, roughly doubling that device's memory.
"""
from __future__ import annotations

import argparse
import gc
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .utils import (LABEL2ID, LABELS, SEED, evaluate_predictions, load_data,
                    safe_wandb_init, set_seed)

MODEL_CANDIDATES = ["microsoft/deberta-v3-large"]


class MCQDataset(Dataset):
    """Wraps pre-tokenised (input_ids, attention_mask) for multiple choice."""

    def __init__(self, ids, mask, labels=None):
        self.ids, self.mask, self.labels = ids, mask, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        item = {"input_ids": self.ids[i], "attention_mask": self.mask[i]}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def encode(df, tokenizer, max_len: int, use_context: bool = False):
    """Tokenise each row into 5 (context, prompt+option) sequences."""
    all_ids, all_mask = [], []
    for _, r in df.iterrows():
        ctx = str(r.get("context", "")) if use_context else ""
        firsts = [ctx] * 5
        seconds = [f"{r['prompt']} {r[c]}" for c in LABELS]
        enc = tokenizer(firsts, seconds, truncation=True,
                        max_length=max_len, padding=False)
        all_ids.append(enc["input_ids"])
        all_mask.append(enc["attention_mask"])
    return all_ids, all_mask


def train_folds(train_df, test_df, model_path, n_folds=5, epochs=2,
                max_len=512, freeze_layers=12, use_context=False, out="artifacts"):
    """Train a K-fold DeBERTa ensemble; return (oof_probs, test_probs)."""
    from transformers import (AutoModelForMultipleChoice, AutoTokenizer,
                              Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tr_ids, tr_mask = encode(train_df, tokenizer, max_len, use_context)
    te_ids, te_mask = encode(test_df, tokenizer, max_len, use_context)

    def collate(batch):
        flat = [{"input_ids": ids, "attention_mask": m}
                for b in batch for ids, m in zip(b["input_ids"], b["attention_mask"])]
        padded = tokenizer.pad(flat, return_tensors="pt")
        out_ = {k: v.view(len(batch), 5, -1) for k, v in padded.items()}
        if "labels" in batch[0]:
            out_["labels"] = torch.tensor([b["labels"] for b in batch])
        return out_

    y = np.array([LABEL2ID[a] for a in train_df["answer"]])
    oof = np.zeros((len(train_df), 5))
    test_probs = np.zeros((len(test_df), 5))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"--- fold {fold + 1}/{n_folds} ---")
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForMultipleChoice.from_pretrained(
            model_path, dtype=torch.float32)
        for p in model.deberta.embeddings.parameters():
            p.requires_grad = False
        for layer in model.deberta.encoder.layer[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False

        args = TrainingArguments(
            output_dir=f"{out}/deberta_f{fold}",
            num_train_epochs=epochs,
            learning_rate=1e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            fp16=torch.cuda.is_available(),
            max_grad_norm=1.0,
            logging_steps=25,
            save_strategy="no",
            report_to="none",
            seed=SEED,
        )
        args._n_gpu = 1

        trainer = Trainer(
            model=model, args=args, data_collator=collate,
            train_dataset=MCQDataset([tr_ids[i] for i in tr_idx],
                                     [tr_mask[i] for i in tr_idx], y[tr_idx]))
        trainer.train()

        val_ds = MCQDataset([tr_ids[i] for i in val_idx],
                            [tr_mask[i] for i in val_idx])
        oof[val_idx] = torch.softmax(
            torch.tensor(trainer.predict(val_ds).predictions), dim=1).numpy()
        test_ds = MCQDataset(te_ids, te_mask)
        test_probs += torch.softmax(
            torch.tensor(trainer.predict(test_ds).predictions), dim=1).numpy() / n_folds

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    return oof, test_probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--use-context", action="store_true",
                    help="feed retrieved RAG context (measured: hurts here)")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed()
    train_df, test_df = load_data()

    oof, test_probs = train_folds(
        train_df, test_df, MODEL_CANDIDATES[0], n_folds=args.folds,
        epochs=args.epochs, max_len=args.max_len,
        use_context=args.use_context, out=args.out)

    np.save(os.path.join(args.out, "deberta_oof_probs.npy"), oof)
    np.save(os.path.join(args.out, "deberta_test_probs.npy"), test_probs)

    metrics = evaluate_predictions(train_df["answer"].values, oof)
    print(f"DeBERTa OOF | MAP@3 {metrics['map3']:.4f} "
          f"| acc {metrics['accuracy']:.4f} | macro-F1 {metrics['f1_macro']:.4f}")

    run = safe_wandb_init(project="24f1002825-t22026", name="deberta-cv")
    if run is not None:
        run.log({"overall_map3": metrics["map3"],
                 "accuracy": metrics["accuracy"],
                 "f1_macro": metrics["f1_macro"]})
        run.finish()


if __name__ == "__main__":
    main()
