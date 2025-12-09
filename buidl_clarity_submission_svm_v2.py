"""
TF-IDF + Linear SVM classifier for Task 2 (evasion)
Simple, stable baseline:
- text = question + " [SEP] " + interview_answer
- Stratified 10% dev split on evasion_label
- TF-IDF word 1-3 grams (max 200k), class_weight balanced, C=1.5
"""

from dataclasses import dataclass
from typing import List

from datasets import DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class Config:
    test_size: float = 0.1
    random_state: int = 42
    submission_file: str = "prediction2"


def load_data(cfg: Config) -> DatasetDict:
    ds = load_dataset("ailsntua/QEvasion")
    for split in ds:
        ds[split] = ds[split].map(
            lambda x: {"text": x["question"] + " [SEP] " + x["interview_answer"]},
        )
    # ensure test order
    ds["test"] = ds["test"].sort("index")
    return ds


def make_splits(ds: DatasetDict, cfg: Config):
    train_indices, dev_indices = train_test_split(
        list(range(len(ds["train"]))),
        test_size=cfg.test_size,
        stratify=ds["train"]["evasion_label"],
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
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=200000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(class_weight="balanced", C=1.5),
            ),
        ]
    )


def eval_and_predict(cfg: Config):
    ds = load_data(cfg)
    train_ds, dev_ds, test_ds = make_splits(ds, cfg)

    model = build_model()
    model.fit(train_ds["text"], train_ds["evasion_label"])

    dev_preds = model.predict(dev_ds["text"])
    acc = accuracy_score(dev_ds["evasion_label"], dev_preds)
    micro = f1_score(dev_ds["evasion_label"], dev_preds, average="micro")
    macro = f1_score(dev_ds["evasion_label"], dev_preds, average="macro")
    print({"dev_accuracy": acc, "dev_micro_f1": micro, "dev_macro_f1": macro})

    test_preds = model.predict(test_ds["text"])
    return test_preds


def write_submission(preds: List[str], path: str):
    assert len(preds) == 308, f"Expected 308 predictions, got {len(preds)}"
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")
    print(f"Wrote {path}")
    print("Zip with: zip prediction2.zip prediction2")


def main():
    cfg = Config()
    preds = eval_and_predict(cfg)
    write_submission(preds, cfg.submission_file)


if __name__ == "__main__":
    main()
