# SemEval-Style Clarity & Evasion Detection (QEvasion / CLARITY)

This repository contains our work toward a SemEval-style system for detecting  
**clarity** and **evasion techniques** in political question–answer pairs.

The project currently uses the [QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion) dataset,  
which includes presidential interview questions and answers annotated for:

- `clarity_label` – high-level clarity of the answer
- `evasion_label` – fine-grained evasion techniques

The goal is to build and evaluate models that can **unmask when a politician dodges a question**.

---

## Project Goals

- Build baseline models for:
  - **Clarity classification** – clear vs. ambiguous/evasive responses.
  - **Fine-grained evasion type classification** – 9+ evasion strategies.
- Explore **transformer-based models** (e.g., RoBERTa / DeBERTa).
- Experiment with **multi-task learning**:
  - Shared encoder with two heads: one for clarity, one for evasion.
- Prepare the codebase to be extended toward a full **SemEval CLARITY task** system.

---

## Project Structure

```text
semeval-clarity-evasion/
├── configs/                  # (later) training & evaluation configs
├── data/                     # local data cache (NOT tracked in git)
├── notebooks/                # EDA & experiments (Jupyter)
├── src/
│   ├── dataset_qevasion.py   # QEvasion dataset loader utilities
│   └── ...                   # (later) models, training & inference scripts
├── requirements.txt
└── README.md
