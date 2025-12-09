"""
Fine-tune DeBERTa-v3-base for CLARITY (Task 1)

Outputs (saved in this folder):
  - bert_dev_proba.npy  (N_dev x 3 softmax probs)
  - bert_test_proba.npy (308 x 3 softmax probs)
  - bert_test_preds.npy (308 label IDs)
  - prediction_deberta (308 labels) for inspection

Shared split file: clarity_split_indices.npz (created here if absent).
Label mapping: Ambivalent->0, Clear Non-Reply->1, Clear Reply->2
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = ["Ambivalent", "Clear Non-Reply", "Clear Reply"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}
BASE_DIR = Path(__file__).resolve().parent
SPLIT_FILE = BASE_DIR / "clarity_split_indices.npz"


@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 256
    learning_rate: float = 1.5e-5
    num_train_epochs: int = 2  # longer to beat SVM
    per_device_train_batch_size: int = 8  # drop to 4 if OOM
    per_device_eval_batch_size: int = 8
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    output_dir: str = str(BASE_DIR / "outputs/deberta_clarity")
    seed: int = 42


def load_data() -> DatasetDict:
    ds = load_dataset("ailsntua/QEvasion")
    for split in ds:
        ds[split] = ds[split].map(
            lambda x: {"text": x["question"] + " [SEP] " + x["interview_answer"]},
        )
    ds["test"] = ds["test"].sort("index")
    return ds


def get_split(ds, test_size=0.1, seed=42):
    from sklearn.model_selection import train_test_split

    if SPLIT_FILE.exists():
        data = np.load(SPLIT_FILE)
        train_idx, dev_idx = data["train_idx"], data["dev_idx"]
    else:
        train_idx, dev_idx = train_test_split(
            range(len(ds["train"])),
            test_size=test_size,
            stratify=ds["train"]["clarity_label"],
            random_state=seed,
        )
        np.savez(SPLIT_FILE, train_idx=train_idx, dev_idx=dev_idx)
    return (
        ds["train"].select(train_idx),
        ds["train"].select(dev_idx),
        ds["test"],
    )


def tokenize_and_label(dataset: DatasetDict, tokenizer, cfg: Config) -> DatasetDict:
    def add_labels(batch):
        return {"labels": [LABEL2ID[lbl] for lbl in batch["clarity_label"]]}

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    labeled = DatasetDict({k: v.map(add_labels, batched=True) for k, v in dataset.items()})
    return DatasetDict({k: v.map(tok, batched=True) for k, v in labeled.items()})


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    cfg = Config()
    set_seed(cfg.seed)

    ds = load_data()
    train_ds, dev_ds, test_ds = get_split(ds, test_size=0.1, seed=cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    tokenized = tokenize_and_label(
        DatasetDict({"train": train_ds, "validation": dev_ds, "test": test_ds}),
        tokenizer,
        cfg,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=50,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    dev_logits = trainer.predict(tokenized["validation"]).predictions
    test_logits = trainer.predict(tokenized["test"]).predictions
    dev_proba = softmax(dev_logits)
    test_proba = softmax(test_logits)

    test_pred_ids = np.argmax(test_proba, axis=1)
    test_preds = [ID2LABEL[i] for i in test_pred_ids]

    np.save(BASE_DIR / "bert_dev_proba.npy", dev_proba)
    np.save(BASE_DIR / "bert_test_proba.npy", test_proba)
    np.save(BASE_DIR / "bert_test_preds.npy", test_pred_ids)

    with open(BASE_DIR / "prediction_deberta", "w", encoding="utf-8") as f:
        for lbl in test_preds:
            f.write(lbl + "\n")

    print("Saved DeBERTa ensemble files in", BASE_DIR)
    print("  bert_dev_proba.npy")
    print("  bert_test_proba.npy")
    print("  bert_test_preds.npy")
    print("Wrote prediction_deberta (308 lines)")


if __name__ == "__main__":
    main()
