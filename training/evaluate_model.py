"""
evaluate_model.py
-----------------
Evaluates the fine-tuned DistilBERT 3-class sentiment model (Experiment 2) on the test split of
the cardiffnlp/tweet_eval (sentiment) dataset.

Calculates:
  - Overall Accuracy
  - Macro Precision, Recall, and F1-Score
  - Per-class Precision, Recall, and F1-Score
  - Detailed Classification Report (NEGATIVE, NEUTRAL, POSITIVE)
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# -----------------------------------------------------------------------------
# Configuration Parameters
# -----------------------------------------------------------------------------
MODEL_DIR = "models/distilbert-sentiment-exp2"  # Path to Experiment 2 model
DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"
NUM_TEST_SAMPLES = 1000  # 1,000 stratified samples for evaluation
MAX_SEQ_LENGTH = 128
# -----------------------------------------------------------------------------

ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
TARGET_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def evaluate():
    print("=" * 60)
    print("EXPERIMENT 2: DISTILBERT MODEL EVALUATION")
    print("=" * 60)

    # 1. Verify Model Directory Exists
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] Fine-tuned model directory '{MODEL_DIR}' does not exist.")
        print("Please run training/train_distilbert.py first.")
        return

    # 2. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device.type.upper()}")

    # 3. Load Saved Model and Tokenizer
    print(f"Loading model & tokenizer from '{MODEL_DIR}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # 4. Load Test Dataset
    print(f"Loading test split from '{DATASET_NAME}' ({DATASET_CONFIG})...")
    test_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")

    if NUM_TEST_SAMPLES and NUM_TEST_SAMPLES < len(test_dataset):
        print(f"Using a stratified evaluation subset of {NUM_TEST_SAMPLES:,} test samples...")
        test_dataset = test_dataset.train_test_split(
            train_size=NUM_TEST_SAMPLES,
            stratify_by_column="label",
            seed=42
        )['train']
    else:
        print(f"Evaluating on full test set ({len(test_dataset):,} samples)...")

    # 5. Model Predictions
    print("Running model inference on test dataset...")
    true_labels = []
    predicted_labels = []

    batch_size = 32
    texts = test_dataset['text']
    labels = test_dataset['label']

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        true_labels.extend(batch_labels)
        predicted_labels.extend(preds)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # 6. Calculate Overall Metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", zero_division=0
    )

    # Calculate Per-Class Metrics
    precisions_per_cls, recalls_per_cls, f1s_per_cls, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0
    )

    # 7. Print Evaluation Results
    print("\n" + "=" * 60)
    print("TEST EVALUATION METRICS:")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro Precision : {precision_macro * 100:.2f}%")
    print(f"Macro Recall    : {recall_macro * 100:.2f}%")
    print(f"Macro F1-Score  : {f1_macro * 100:.2f}%")

    print("\n" + "=" * 60)
    print("PER-CLASS METRICS SUMMARY:")
    print("=" * 60)
    for idx, name in enumerate(TARGET_NAMES):
        print(f"Class {idx} ({name:<8}): Precision={precisions_per_cls[idx]*100:.2f}% | Recall={recalls_per_cls[idx]*100:.2f}% | F1={f1s_per_cls[idx]*100:.2f}%")

    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT:")
    print("=" * 60)
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0
    )
    print(report)
    print("=" * 60)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precisions_per_cls": precisions_per_cls,
        "recalls_per_cls": recalls_per_cls,
        "f1s_per_cls": f1s_per_cls,
        "report": report
    }


if __name__ == "__main__":
    evaluate()
