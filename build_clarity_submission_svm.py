"""
Train a TF-IDF + Linear SVM classifier for CLARITY (Task 1) and write a submission file.

Steps:
- Load QEvasion.
- text = question + " [SEP] " + interview_answer.
- Stratified 10% dev split on clarity_label.
- Train TF-IDF (1-3 grams) + LinearSVC (class_weight='balanced').
- Report accuracy, micro_f1, macro_f1 on dev.
- Predict the 308-row test split in original order and write `prediction_svm`.
- Print zip command.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


LABELS = ["Ambivalent", "Clear Non-Reply", "Clear Reply"]


@dataclass
class Config:
    test_size: float = 0.1
    random_state: int = 42
    submission_file: str = "prediction_svm"


def load_data(cfg: Config) -> DatasetDict:
    ds = load_dataset("ailsntua/QEvasion")
    for split in ds:
        ds[split] = ds[split].map(
            lambda x: {"text": x["question"] + " [SEP] " + x["interview_answer"]},
        )
    # ensure test in order
    ds["test"] = ds["test"].sort("index")
    return ds


def make_splits(ds: DatasetDict, cfg: Config):
    train_indices, dev_indices = train_test_split(
        list(range(len(ds["train"]))),
        test_size=cfg.test_size,
        stratify=ds["train"]["clarity_label"],
        random_state=cfg.random_state,
    )
    train = ds["train"].select(train_indices)
    dev = ds["train"].select(dev_indices)
    return train, dev, ds["test"]


def build_model():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=150000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(class_weight="balanced"),
            ),
        ]
    )


def eval_and_predict(cfg: Config):
    ds = load_data(cfg)
    train_ds, dev_ds, test_ds = make_splits(ds, cfg)

    model = build_model()
    model.fit(train_ds["text"], train_ds["clarity_label"])

    dev_preds = model.predict(dev_ds["text"])
    acc = accuracy_score(dev_ds["clarity_label"], dev_preds)
    micro = f1_score(dev_ds["clarity_label"], dev_preds, average="micro")
    macro = f1_score(dev_ds["clarity_label"], dev_preds, average="macro")
    print({"dev_accuracy": acc, "dev_micro_f1": micro, "dev_macro_f1": macro})

    test_preds = model.predict(test_ds["text"])
    return test_preds


def write_submission(preds: List[str], path: str):
    assert len(preds) == 308, f"Expected 308 predictions, got {len(preds)}"
    assert set(preds).issubset(set(LABELS)), f"Unexpected labels: {set(preds) - set(LABELS)}"
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")
    print(f"Wrote {path}")
    print("Zip with: zip prediction.zip prediction_svm")


def main():
    cfg = Config()
    preds = eval_and_predict(cfg)
    write_submission(preds, cfg.submission_file)


if __name__ == "__main__":
    main()
