"""
prepare_dataset.py
------------------
Loads and inspects the cardiffnlp/tweet_eval (sentiment) dataset from Hugging Face.
Reports split statistics, class distributions, label mappings, and creates a small
sample CSV for inspection.
"""

import os
import pandas as pd
from collections import Counter
from datasets import load_dataset


def inspect_dataset():
    print("=" * 60)
    print("DAY 1 STEP 3: DATASET INSPECTION (cardiffnlp/tweet_eval - sentiment)")
    print("=" * 60)

    try:
        print("\nLoading dataset 'cardiffnlp/tweet_eval' with config 'sentiment'...")
        dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
        print("Dataset loaded successfully!\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        raise e

    # 1. Label Mapping Verification
    label_feature = dataset['train'].features['label']
    label_names = label_feature.names if hasattr(label_feature, 'names') else ['negative', 'neutral', 'positive']
    label_mapping = {i: name for i, name in enumerate(label_names)}

    print("LABEL MAPPING METADATA:")
    for num_label, name in label_mapping.items():
        print(f"  {num_label} -> {name.upper()}")
    print("-" * 60)

    # 2. Split Sizes and Class Distributions
    print("\nDATASET SPLIT STATISTICS & CLASS DISTRIBUTIONS:")
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset[split_name]
        total_count = len(split_data)
        labels = split_data['label']
        distribution = Counter(labels)

        print(f"\n--- {split_name.upper()} SPLIT (Total: {total_count:,} samples) ---")
        for num_label, name in label_mapping.items():
            count = distribution.get(num_label, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            print(f"  Label {num_label} ({name.upper():<8}): {count:>6,} samples ({percentage:.2f}%)")

    # 3. Print 5 Example Records
    print("\n" + "=" * 60)
    print("5 EXAMPLE RECORDS (from Train Split):")
    print("=" * 60)
    for idx in range(min(5, len(dataset['train']))):
        item = dataset['train'][idx]
        lbl = item['label']
        lbl_name = label_mapping.get(lbl, 'unknown').upper()
        text = item['text'].replace('\n', ' ')
        print(f"Example {idx + 1}:")
        print(f"  Text         : \"{text}\"")
        print(f"  Numeric Label: {lbl}")
        print(f"  Sentiment    : {lbl_name}\n")

    # 4. Create Small Inspection Sample CSV (data/sample_sentiment.csv)
    os.makedirs("data", exist_ok=True)
    sample_filepath = os.path.join("data", "sample_sentiment.csv")

    # Gather 5 examples of each class from validation set for inspection sample
    sample_records = []
    samples_per_class = {0: 0, 1: 0, 2: 0}
    target_per_class = 5

    for item in dataset['validation']:
        lbl = item['label']
        if samples_per_class.get(lbl, 0) < target_per_class:
            sample_records.append({
                "text": item['text'].replace('\n', ' '),
                "label": lbl,
                "sentiment": label_mapping.get(lbl, 'unknown').upper()
            })
            samples_per_class[lbl] += 1
        if all(c >= target_per_class for c in samples_per_class.values()):
            break

    df_sample = pd.DataFrame(sample_records)
    df_sample.to_csv(sample_filepath, index=False)
    print(f"Created small inspection sample file at: {sample_filepath} ({len(df_sample)} rows)")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
