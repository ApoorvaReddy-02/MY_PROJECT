"""
test_sentiment.py
-----------------
Test script to evaluate sentiment_model.py on sample text inputs.
"""

from sentiment.sentiment_model import analyze_sentiment, MODEL_NAME

def main():
    print("=" * 60)
    print("DAY 1 STEP 2: DISTILBERT SENTIMENT ANALYSIS TEST")
    print("=" * 60)
    print(f"Pretrained Model Used: {MODEL_NAME}")
    print("Model Limitation Note: SST-2 DistilBERT model natively supports binary classification (POSITIVE/NEGATIVE only). It does not include a discrete NEUTRAL class.\n")

    test_inputs = [
        "I absolutely love this phone. The camera is amazing.",
        "The battery life is terrible and the phone gets very hot.",
        "The product is okay."
    ]

    for idx, sample_text in enumerate(test_inputs, 1):
        result = analyze_sentiment(sample_text)
        print(f"Example {idx}:")
        print(f"  Input Text: \"{sample_text}\"")
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['score'] * 100:.2f}%\n")

if __name__ == "__main__":
    main()
