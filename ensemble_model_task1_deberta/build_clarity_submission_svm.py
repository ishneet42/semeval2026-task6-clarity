"""
TF-IDF + Linear SVM for CLARITY (Task 1) with optional ensembling using
the local DeBERTa artifacts in this folder.

Run order:
1) .venv/bin/python build_clarity_submission_deberta.py
   -> saves clarity_split_indices.npz, bert_*_proba.npy, bert_*_preds.npy
2) .venv/bin/python build_clarity_submission_svm.py
   -> trains SVM, compares ensemble vs SVM-only on dev, writes prediction_svm
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
SPLIT_FILE = BASE_DIR / "clarity_split_indices.npz"


@dataclass
class Config:
    test_size: float = 0.1
    random_state: int = 42
    submission_file: str = "prediction_svm"
    use_ensemble: bool = True
    svm_c: float = 1.5
    svm_max_features: int = 150_000
    ensemble_dir: Path = BASE_DIR  # where bert_* files live


def parse_args(cfg: Config):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-file", default=cfg.submission_file)
    parser.add_argument("--svm-c", type=float, default=cfg.svm_c)
    parser.add_argument("--svm-max-features", type=int, default=cfg.svm_max_features)
    parser.add_argument("--ensemble-dir", default=str(cfg.ensemble_dir))
    args = parser.parse_args()

    cfg.submission_file = args.submission_file
    cfg.svm_c = args.svm_c
    cfg.svm_max_features = args.svm_max_features
    cfg.ensemble_dir = Path(args.ensemble_dir)
    return cfg


def load_data() -> DatasetDict:
    ds = load_dataset("ailsntua/QEvasion")
    for split in ds:
        ds[split] = ds[split].map(
            lambda x: {"text": x["question"] + " [SEP] " + x["interview_answer"]},
        )
    ds["test"] = ds["test"].sort("index")
    return ds


def get_split(ds, cfg: Config):
    if SPLIT_FILE.exists():
        data = np.load(SPLIT_FILE)
        train_idx, dev_idx = data["train_idx"], data["dev_idx"]
    else:
        train_idx, dev_idx = train_test_split(
            range(len(ds["train"])),
            test_size=cfg.test_size,
            stratify=ds["train"]["clarity_label"],
            random_state=cfg.random_state,
        )
        np.savez(SPLIT_FILE, train_idx=train_idx, dev_idx=dev_idx)
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
    return np.load(paths[0]), np.load(paths[1]), np.load(paths[2])


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
    cfg = parse_args(Config())
    ds = load_data()
    train_ds, dev_ds, test_ds = get_split(ds, cfg)

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
