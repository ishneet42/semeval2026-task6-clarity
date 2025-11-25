from datasets import load_dataset

dataset = load_dataset("ailsntua/QEvasion", split="train")

dataset = dataset.remove_columns(["title", "date", "url", "annotator_id", "annotator1", "annotator2","annotator3"])
Clarity = df["Clear Reply", "Clear Non-reply", "Ambivalent Reply"]

print(dataset.column_names)