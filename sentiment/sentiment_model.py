"""
sentiment_model.py
------------------
Module for performing sentiment analysis using a pre-trained DistilBERT model.
Uses Hugging Face Transformers pipeline for text classification.
"""

from transformers import pipeline

# Load pretrained DistilBERT sentiment analysis pipeline
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Note on Model Limitation:
# The SST-2 finetuned DistilBERT model supports 2 sentiment classes: POSITIVE and NEGATIVE.
# It does not natively support a NEUTRAL class.

print(f"Loading Hugging Face model '{MODEL_NAME}'...")
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)


def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of the provided text string.

    Args:
        text (str): Input text to analyze.

    Returns:
        dict: Dictionary containing:
            - 'label': Predicted sentiment label ('POSITIVE' or 'NEGATIVE')
            - 'score': Confidence score of the prediction (float between 0.0 and 1.0)
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {"label": "UNKNOWN", "score": 0.0}

    results = sentiment_pipeline(text)
    prediction = results[0]

    return {
        "label": prediction["label"],
        "score": float(prediction["score"])
    }
