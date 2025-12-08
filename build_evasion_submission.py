from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_data():
    dataset = load_dataset("ailsntua/QEvasion")
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_df["text"] = train_df["question"] + " [SEP] " + train_df["interview_answer"]
    test_df["text"] = test_df["question"] + " [SEP] " + test_df["interview_answer"]

    test_df = test_df.sort_values("index").reset_index(drop=True)
    return train_df, test_df


def train_model(train_df):
    clf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    clf.fit(train_df["text"], train_df["evasion_label"])
    return clf


def write_submission(preds):
    allowed = {
        "Claims ignorance",
        "Clarification",
        "Declining to answer",
        "Deflection",
        "Dodging",
        "Explicit",
        "General",
        "Implicit",
        "Partial/half-answer",
    }
    assert len(preds) == 308, f"Expected 308 predictions, got {len(preds)}"
    assert set(preds).issubset(allowed), f"Unexpected labels: {set(preds) - allowed}"

    with open("prediction2", "w", encoding="utf-8") as f:
        for label in preds:
            f.write(label + "\n")

    print(f"Wrote file 'prediction2' with {len(preds)} lines")
    print("Zip for Codabench with: zip prediction2.zip prediction2")


def main():
    train_df, test_df = load_data()
    model = train_model(train_df)
    preds = model.predict(test_df["text"]).tolist()
    write_submission(preds)


if __name__ == "__main__":
    main()
