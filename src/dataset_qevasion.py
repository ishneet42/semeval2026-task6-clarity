from datasets import load_dataset

def load_qevasion(split: str = "train"):
    
    # Load the QEvasion dataset split from Hugging Face.
    
    dataset = load_dataset("ailsntua/QEvasion", split=split)
    return dataset

if __name__ == "__main__":
    ds = load_qevasion("train")
    print(ds)
    print(ds[0]["interview_question"])
    print(ds[0]["interview_answer"])
    print(ds[0]["clarity_label"], ds[0]["evasion_label"])