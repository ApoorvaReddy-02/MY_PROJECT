"""
reddit_connector.py
-------------------
Reddit Dataset Connector for the Data Collection Agent.

Enforces strict relevance filtering on Reddit_Data.csv.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from .base_connector import BaseConnector


class RedditConnector(BaseConnector):
    """Reddit dataset connector for sentiment analysis."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or os.path.join(
            "datasets", "Reddit_Data.csv"
        )

        super().__init__(
            source_name="reddit",
            api_key=None
        )

    def fetch_data(
        self,
        query: str = "",
        max_items: int = 10
    ) -> List[Dict[str, Any]]:

        query = str(query or "").strip()
        if not query:
            return []

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"[RedditConnector] Dataset not found: {self.dataset_path}"
            )

        df = pd.read_csv(self.dataset_path)
        df = df.dropna(subset=["clean_comment"])

        # Token-based relevance filtering
        query_clean = query.lower()
        raw_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", query_clean)
        stopwords = {
            "a", "an", "the", "and", "or", "for", "is", "of", "in", "on", "to",
            "with", "about", "at", "by", "from", "it", "this", "that"
        }
        tokens = [t for t in raw_tokens if t not in stopwords and len(t) > 1]
        if not tokens:
            tokens = raw_tokens

        mask = pd.Series(True, index=df.index)
        if tokens:
            for token in tokens:
                mask = mask & df["clean_comment"].astype(str).str.lower().str.contains(token, na=False, regex=False)
        else:
            mask = df["clean_comment"].astype(str).str.lower().str.contains(query_clean, na=False, regex=False)

        matching = df[mask]
        matching = matching.head(max_items)

        results = []
        for _, row in matching.iterrows():
            try:
                category = int(float(row["category"]))
                if category == 1:
                    sentiment = "positive"
                elif category == -1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
            except (ValueError, TypeError):
                sentiment = "neutral"

            results.append({
                "source": "reddit",
                "title": "Reddit Comment",
                "text": str(row["clean_comment"]),
                "url": "",
                "timestamp": "",
                "rating": "",
                "category": sentiment,
                "product_name": ""
            })

        print(f"[RedditConnector] Dataset search for '{query}' complete. Matching records retrieved: {len(results)}")
        return results