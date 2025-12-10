"""
DeBERTa-v3-base for CLARITY (Task 1) 
Steps:
1. Load the QEvasion dataset.
2. Build text = question + " [SEP] " + interview_answer.
3. Stratified 80/20 train/dev split on clarity_label.
4. Fine-tune microsoft/deberta-v3-base with specified hyperparameters.
5. Report accuracy, micro_f1, macro_f1 on dev; warn if micro_f1 < baseline (~0.65).
6. Predict the test split (308 rows, original order), map ids to labels, and write `prediction`.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List

import argparse
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Fixed label mapping
LABEL2ID: Dict[str, int] = {
    "Ambivalent": 0,
    "Clear Non-Reply": 1,
    "Clear Reply": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

BASELINE_MICRO_F1 = 0.65  # TF-IDF reference


@dataclass
class Config:
    model_name: str = "roberta-base"
    max_length: int = 192
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "outputs/roberta_clarity"
    seed: int = 42
    submission_file: str = "prediction"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-epochs", type=float, default=3, help="Override number of epochs")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Override per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override per-device eval batch size")
    parser.add_argument("--max-length", type=int, default=None, help="Override max sequence length")
    return parser.parse_args()


def load_and_prepare(cfg: Config) -> DatasetDict:
    raw = load_dataset("ailsntua/QEvasion")
    for split in raw:
        raw[split] = raw[split].map(
            lambda x: {"text": x["question"] + " [SEP] " + x["interview_answer"]},
            remove_columns=[],
        )
    train_indices, val_indices = train_test_split(
        list(range(len(raw["train"]))),
        test_size=0.2,
        stratify=raw["train"]["clarity_label"],
        random_state=cfg.seed,
    )
    train_split = raw["train"].select(train_indices)
    val_split = raw["train"].select(val_indices)
    test_split = raw["test"].sort("index")
    return DatasetDict({"train": train_split, "validation": val_split, "test": test_split})


def tokenize_and_label(dataset: DatasetDict, tokenizer: AutoTokenizer, cfg: Config) -> DatasetDict:
    def add_labels(batch):
        return {"labels": [LABEL2ID[lbl] for lbl in batch["clarity_label"]]}

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    labeled = DatasetDict({k: v.map(add_labels, batched=True) for k, v in dataset.items()})
    tokenized = DatasetDict({k: v.map(tokenize, batched=True) for k, v in labeled.items()})
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    args = parse_args()
    cfg = Config(
        num_train_epochs=args.num_train_epochs,
        output_dir=args.output_dir or Config.output_dir,
        per_device_train_batch_size=args.train_batch_size or Config.per_device_train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size or Config.per_device_eval_batch_size,
        max_length=args.max_length or Config.max_length,
    )
    set_seed(cfg.seed)

    data = load_and_prepare(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    tokenized = tokenize_and_label(data, tokenizer, cfg)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="epoch",
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
    eval_metrics = trainer.evaluate()
    micro = eval_metrics.get("eval_micro_f1", 0.0)
    print(f"Dev metrics: {eval_metrics}")
    if micro < BASELINE_MICRO_F1:
        print(f"WARNING: Dev micro_f1 {micro:.4f} is below TF-IDF baseline {BASELINE_MICRO_F1:.2f}")

    # Predict test with best checkpoint
    test_preds = trainer.predict(tokenized["test"]).predictions
    pred_ids = np.argmax(test_preds, axis=-1)
    preds = [ID2LABEL[i] for i in pred_ids]

    # Submission file
    assert len(preds) == 308, f"Expected 308 predictions, got {len(preds)}"
    allowed = set(LABEL2ID.keys())
    assert set(preds).issubset(allowed), f"Unexpected labels: {set(preds) - allowed}"

    with open(cfg.submission_file, "w", encoding="utf-8") as f:
        for label in preds:
            f.write(label + "\n")

    print("Wrote prediction")
    print("Zip with: zip prediction.zip prediction")


if __name__ == "__main__":
    main()
