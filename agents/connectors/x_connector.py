"""
x_connector.py
--------------
X (Twitter) Data Source Connector.

Supports:
1. Live X API when X_BEARER_TOKEN is available.
2. Twitter_Data.csv dataset when API is unavailable.

Enforces strict relevance filtering: Never returns fallback/unrelated tweets if no match is found.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from .base_connector import BaseConnector


class XConnector(BaseConnector):
    """X / Twitter connector with API + dataset relevance search."""

    def __init__(self, bearer_token: Optional[str] = None):
        bearer_token = (
            bearer_token
            or os.getenv("X_BEARER_TOKEN")
            or os.getenv("TWITTER_BEARER_TOKEN")
        )

        super().__init__(source_name="x", api_key=bearer_token)

        self.dataset_path = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            ),
            "datasets",
            "Twitter_Data.csv"
        )

    def fetch_data(
        self,
        query: str,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:

        query = str(query or "").strip()

        # ============================================================
        # OPTION 1: LIVE X API
        # ============================================================
        if self.api_key:
            try:
                import tweepy

                client = tweepy.Client(bearer_token=self.api_key)

                response = client.search_recent_tweets(
                    query=query,
                    max_results=max(10, min(max_items, 100)),
                    tweet_fields=["created_at"]
                )

                results = []
                if response.data:
                    for tweet in response.data[:max_items]:
                        results.append({
                            "source": "x",
                            "title": "X Post",
                            "text": tweet.text,
                            "url": f"https://x.com/i/web/status/{tweet.id}",
                            "timestamp": str(tweet.created_at) if tweet.created_at else "",
                            "rating": "",
                            "category": "",
                            "product_name": ""
                        })

                return results

            except Exception as e:
                print(f"[XConnector] Live API failed: {e}")
                print("[XConnector] Falling back to Twitter dataset...")

        # ============================================================
        # OPTION 2: DATASET SEARCH (STRICT RELEVANCE)
        # ============================================================
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"[XConnector] Twitter dataset not found:\n{self.dataset_path}"
            )

        if not query:
            return []

        df = pd.read_csv(self.dataset_path)

        required_columns = {"clean_text", "category"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                "[XConnector] Dataset must contain 'clean_text' and 'category' columns."
            )

        df = df.dropna(subset=["clean_text"])

        # Token-based relevance search
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
                mask = mask & df["clean_text"].astype(str).str.lower().str.contains(token, na=False, regex=False)
        else:
            mask = df["clean_text"].astype(str).str.lower().str.contains(query_clean, na=False, regex=False)

        matching = df[mask]

        # Return matching records up to max_items. Do NOT fall back to full dataset if len(matching) == 0.
        matching = matching.head(max_items)

        results = []
        for _, row in matching.iterrows():
            category = row["category"]
            try:
                category_num = int(float(category))
                if category_num == 1:
                    sentiment = "positive"
                elif category_num == -1:
                    sentiment = "negative"
                elif category_num == 0:
                    sentiment = "neutral"
                else:
                    sentiment = "neutral"
            except (ValueError, TypeError):
                sentiment = str(category)

            results.append({
                "source": "x",
                "title": "Twitter/X Post",
                "text": str(row["clean_text"]),
                "url": "",
                "timestamp": "",
                "rating": "",
                "category": sentiment,
                "product_name": ""
            })

        print(f"[XConnector] Dataset search for '{query}' complete. Matching records retrieved: {len(results)}")
        return results