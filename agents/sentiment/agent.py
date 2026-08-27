"""
Sentiment Analysis Agent
------------------------
Uses a pretrained DistilBERT model to analyze the sentiment
of records produced by the Preprocessing Agent.

Input:
    List[Dict] containing a 'cleaned_text' field

Output:
    Same records with:
        - sentiment
        - sentiment_score
"""

from typing import Any, Dict, List

from transformers import pipeline


class SentimentAnalysisAgent:
    """
    Performs sentiment analysis using a pretrained DistilBERT model.
    """

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self) -> None:
        print(
            f"[SentimentAnalysisAgent] Loading model: {self.MODEL_NAME}"
        )

        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
        )

        print("[SentimentAnalysisAgent] Model loaded successfully.")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment for a single piece of text.

        Returns:
            {
                "sentiment": "POSITIVE" or "NEGATIVE",
                "sentiment_score": float
            }
        """

        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "UNKNOWN",
                "sentiment_score": 0.0,
            }

        result = self.classifier(text, truncation=True)[0]

        return {
            "sentiment": result["label"],
            "sentiment_score": float(result["score"]),
        }

    def process(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process records received from the Preprocessing Agent.

        Original fields are preserved.
        Sentiment information is added to each record.
        """

        if not isinstance(records, list):
            return []

        processed_records: List[Dict[str, Any]] = []

        for record in records:
            if not isinstance(record, dict):
                continue

            cleaned_text = record.get("cleaned_text", "")

            if not isinstance(cleaned_text, str) or not cleaned_text.strip():
                continue

            sentiment_result = self.analyze_text(cleaned_text)

            processed_record = record.copy()

            processed_record["sentiment"] = sentiment_result["sentiment"]
            processed_record["sentiment_score"] = sentiment_result[
                "sentiment_score"
            ]

            processed_records.append(processed_record)

        return processed_records