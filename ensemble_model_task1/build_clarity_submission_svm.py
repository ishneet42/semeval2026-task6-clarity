"""
TF-IDF + Linear SVM for CLARITY (Task 1) with optional ensembling.

BERT/DeBERTa artifacts are loaded from a sibling folder so you can compare:
  - ensemble_model_task1/ (for the BERT baseline)
  - ensemble_model_task1_deberta/ (for the DeBERTa variant)

Run the chosen BERT/DeBERTa script first to generate:
  bert_dev_proba.npy
  bert_test_proba.npy
  bert_test_preds.npy
  clarity_split_indices.npz  (shared split)

Then run this script to train SVM, ensemble, and write `prediction_svm`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABELS = ["Ambivalent", "Clear Non-Reply", "Clear Reply"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    test_size: float = 0.1
    random_state: int = 42
    submission_file: str = "prediction_svm"
    use_ensemble: bool = True
    svm_c: float = 1.5
    svm_max_features: int = 150_000
    # where to look for BERT/DeBERTa files
    ensemble_dir: Path = BASE_DIR  # adjust to ensemble_model_task1_deberta if desired


def load_data() -> DatasetDict:
    ds = load_dataset("ailsntua/QEvasion")
    for split in ds:
        ds[split] = ds[split].map(
            lambda x: {"text": x["question"] + " [SEP] " + x["interview_answer"]},
        )
    ds["test"] = ds["test"].sort("index")
    return ds


def get_split(ds, cfg: Config, split_file: Path):
    if split_file.exists():
        data = np.load(split_file)
        train_idx, dev_idx = data["train_idx"], data["dev_idx"]
    else:
        train_idx, dev_idx = train_test_split(
            range(len(ds["train"])),
            test_size=cfg.test_size,
            stratify=ds["train"]["clarity_label"],
            random_state=cfg.random_state,
        )
        np.savez(split_file, train_idx=train_idx, dev_idx=dev_idx)
    return (
        ds["train"].select(train_idx),
        ds["train"].select(dev_idx),
        ds["test"],
    )


def build_svm(cfg: Config):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=cfg.svm_max_features,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced", C=cfg.svm_c)),
        ]
    )


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_ensemble_files(cfg: Config):
    paths = [
        cfg.ensemble_dir / "bert_dev_proba.npy",
        cfg.ensemble_dir / "bert_test_proba.npy",
        cfg.ensemble_dir / "bert_test_preds.npy",
    ]
    if not all(p.exists() for p in paths):
        print("Ensemble skipped: files not found in", cfg.ensemble_dir)
        return None, None, None
    return (
        np.load(paths[0]),
        np.load(paths[1]),
        np.load(paths[2]),
    )


def ensemble(y_dev, svm_dev_proba, svm_test_proba, bert_dev_proba, bert_test_proba):
    # Soft voting
    soft_dev = 0.6 * svm_dev_proba + 0.4 * bert_dev_proba
    soft_dev_preds = np.argmax(soft_dev, axis=1)

    # Stacking
    X_dev = np.concatenate([svm_dev_proba, bert_dev_proba], axis=1)
    X_test = np.concatenate([svm_test_proba, bert_test_proba], axis=1)
    meta = LogisticRegression(max_iter=500, multi_class="auto")
    meta.fit(X_dev, y_dev)
    stack_dev_preds = meta.predict(X_dev)
    stack_test_preds = meta.predict(X_test)

    # SVM-only
    svm_dev_preds = np.argmax(svm_dev_proba, axis=1)

    # Pick best by dev micro, then macro
    candidates = [
        ("svm", f1_score(y_dev, svm_dev_preds, average="micro"), f1_score(y_dev, svm_dev_preds, average="macro"), np.argmax(svm_test_proba, axis=1)),
        ("soft", f1_score(y_dev, soft_dev_preds, average="micro"), f1_score(y_dev, soft_dev_preds, average="macro"), np.argmax(0.6 * svm_test_proba + 0.4 * bert_test_proba, axis=1)),
        ("stack", f1_score(y_dev, stack_dev_preds, average="micro"), f1_score(y_dev, stack_dev_preds, average="macro"), stack_test_preds),
    ]
    best = max(candidates, key=lambda x: (x[1], x[2]))
    print("Ensemble choice:", best[0], "dev_micro:", best[1], "dev_macro:", best[2])
    return best[3]


def write_submission(preds: List[int], path: str):
    assert len(preds) == 308, f"Expected 308 predictions, got {len(preds)}"
    labels = [ID2LABEL[int(i)] for i in preds]
    with open(path, "w", encoding="utf-8") as f:
        for lbl in labels:
            f.write(lbl + "\n")
    print(f"Wrote {path}")
    print("Zip with: zip prediction.zip prediction_svm")


if __name__ == "__main__":
    cfg = Config()
    ds = load_data()
    split_file = cfg.ensemble_dir / "clarity_split_indices.npz"
    train_ds, dev_ds, test_ds = get_split(ds, cfg, split_file)

    train_texts, train_labels = train_ds["text"], train_ds["clarity_label"]
    dev_texts, dev_labels = dev_ds["text"], dev_ds["clarity_label"]
    test_texts = test_ds["text"]

    model = build_svm(cfg)
    model.fit(train_texts, train_labels)

    dev_preds = model.predict(dev_texts)
    acc = accuracy_score(dev_labels, dev_preds)
    micro = f1_score(dev_labels, dev_preds, average="micro")
    macro = f1_score(dev_labels, dev_preds, average="macro")
    print({"dev_accuracy": acc, "dev_micro_f1": micro, "dev_macro_f1": macro})

    dev_logits = model.decision_function(dev_texts)
    test_logits = model.decision_function(test_texts)
    svm_dev_proba = softmax(dev_logits)
    svm_test_proba = softmax(test_logits)

    bert_dev_proba, bert_test_proba, bert_test_preds = load_ensemble_files(cfg)

    if cfg.use_ensemble and bert_dev_proba is not None:
        y_dev_ids = np.array([LABEL2ID[l] for l in dev_labels])
        final_preds = ensemble(
            y_dev_ids,
            svm_dev_proba,
            svm_test_proba,
            bert_dev_proba,
            bert_test_proba,
        )
    else:
        final_preds = np.argmax(svm_test_proba, axis=1)

    write_submission(final_preds, cfg.submission_file)
