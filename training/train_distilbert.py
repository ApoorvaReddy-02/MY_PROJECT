"""
train_distilbert.py
-------------------
Experiment 2:
Fine-tunes distilbert-base-uncased on cardiffnlp/tweet_eval (sentiment) for 3-class classification:
  0 = NEGATIVE
  1 = NEUTRAL
  2 = POSITIVE

Includes:
  - 6,000 stratified training samples
  - 300 stratified validation samples
  - 3 training epochs
  - Dynamically calculated inverse class weights applied via custom WeightedTrainer (CrossEntropyLoss)
  - Saves model to models/distilbert-sentiment-exp2/
"""

import os
import time
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------------------------------------------------------
# Configuration Parameters (Hyperparameters for Experiment 2)
# -----------------------------------------------------------------------------
BASE_MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"

NUM_TRAIN_SAMPLES = 6000   # 6,000 stratified samples for training
NUM_VAL_SAMPLES = 300      # 300 stratified samples for validation
NUM_EPOCHS = 3             # 3 training epochs
BATCH_SIZE = 16            # Batch size per device
LEARNING_RATE = 2e-5       # Learning rate for AdamW optimizer
MAX_SEQ_LENGTH = 128       # Max sequence length for tokenization
OUTPUT_MODEL_DIR = "models/distilbert-sentiment-exp2"  # Experiment 2 model path
# -----------------------------------------------------------------------------

# Label Definitions
ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}


class WeightedTrainer(Trainer):
    """Custom Hugging Face Trainer that incorporates class weights into CrossEntropyLoss."""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Calculates accuracy, precision, recall, and F1 score for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def train():
    print("=" * 60)
    print("EXPERIMENT 2: DISTILBERT 3-CLASS FINE-TUNING (CLASS-WEIGHTED)")
    print("=" * 60)

    # 1. Device Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device Status: Running on {device.upper()}")
    if device == "cpu":
        print("  -> GPU not detected. Training will run on CPU using conservative settings.")
    else:
        print(f"  -> GPU detected: {torch.cuda.get_device_name(0)}")

    # 2. Load Dataset
    print(f"\nLoading dataset '{DATASET_NAME}' ({DATASET_CONFIG})...")
    raw_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # 3. Stratified Subset Selection
    print(f"Creating stratified subsets:")
    print(f"  Training samples  : {NUM_TRAIN_SAMPLES:,}")
    print(f"  Validation samples: {NUM_VAL_SAMPLES:,}")

    train_subset = raw_dataset['train'].train_test_split(
        train_size=NUM_TRAIN_SAMPLES,
        stratify_by_column="label",
        seed=42
    )['train']

    val_subset = raw_dataset['validation'].train_test_split(
        train_size=NUM_VAL_SAMPLES,
        stratify_by_column="label",
        seed=42
    )['train']

    # 4. Calculate Class Weights dynamically from training subset
    train_labels = np.array(train_subset['label'])
    label_counts = Counter(train_labels)
    classes = np.unique(train_labels)

    raw_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float)

    print("\nTRAINING SUBSET CLASS DISTRIBUTION & WEIGHTS:")
    for cls_idx in classes:
        count = label_counts[cls_idx]
        weight = raw_weights[cls_idx]
        print(f"  Class {cls_idx} ({ID2LABEL[cls_idx]:<8}): {count:>5,} samples | Loss Weight = {weight:.4f}")

    # 5. Tokenization
    print(f"\nLoading tokenizer '{BASE_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )

    print("Tokenizing datasets...")
    tokenized_train = train_subset.map(preprocess_function, batched=True)
    tokenized_val = val_subset.map(preprocess_function, batched=True)

    # 6. Load Pretrained Model
    print(f"Loading base model '{BASE_MODEL_NAME}' with 3 output labels...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 7. Define Training Arguments
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir="./results_temp_exp2",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        use_cpu=(device == "cpu")
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 8. Execute Training
    print("\nStarting model training for Experiment 2...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")

    # 9. Evaluate on Validation Set
    print("\nEvaluating model on validation subset...")
    eval_metrics = trainer.evaluate()
    print("Validation Results:")
    print(f"  Loss     : {eval_metrics.get('eval_loss', 0.0):.4f}")
    print(f"  Accuracy : {eval_metrics.get('eval_accuracy', 0.0) * 100:.2f}%")
    print(f"  F1-Score : {eval_metrics.get('eval_f1', 0.0) * 100:.2f}%")
    print(f"  Precision: {eval_metrics.get('eval_precision', 0.0) * 100:.2f}%")
    print(f"  Recall   : {eval_metrics.get('eval_recall', 0.0) * 100:.2f}%")

    # 10. Save Model and Tokenizer
    print(f"\nSaving fine-tuned model & tokenizer to '{OUTPUT_MODEL_DIR}'...")
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print("Model and tokenizer saved successfully!")
    print("=" * 60)

    return {
        "success": True,
        "elapsed_time": elapsed_time,
        "eval_metrics": eval_metrics,
        "class_weights": raw_weights
    }


if __name__ == "__main__":
    train()
