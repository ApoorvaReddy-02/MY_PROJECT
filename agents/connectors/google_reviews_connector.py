"""
google_reviews_connector.py
---------------------------
Google Reviews Dataset Connector.

Uses a local Google Reviews dataset.
No Google API or billing is required.

Dataset columns:
    review
    rating
    date
    business
    location
    source
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional

from .base_connector import BaseConnector


class GoogleReviewsConnector(BaseConnector):
    """Local Google Reviews dataset connector."""

    def __init__(self, dataset_path: Optional[str] = None):

        self.dataset_path = dataset_path or os.path.join(
            "datasets",
            "google_reviews_dataset.csv"
        )

        super().__init__(
            source_name="google",
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
                f"[GoogleReviewsConnector] Dataset not found: "
                f"{self.dataset_path}"
            )

        print("[GoogleReviewsConnector] Searching local Google Reviews dataset...")
        print(f"Query: {query}")

        # ---------------------------------------------------------
        # Load dataset
        # ---------------------------------------------------------
        df = pd.read_csv(self.dataset_path)

        required_columns = [
            "review",
            "rating",
            "date",
            "business",
            "location",
            "source"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "[GoogleReviewsConnector] Missing columns: "
                + ", ".join(missing_columns)
            )

        # Remove rows without review text
        df = df.dropna(subset=["review"])

        # Convert searchable fields to strings
        df["review"] = df["review"].fillna("").astype(str)
        df["business"] = df["business"].fillna("").astype(str)
        df["location"] = df["location"].fillna("").astype(str)

        # ---------------------------------------------------------
        # Build searchable text
        # ---------------------------------------------------------
        df["_search_text"] = (
            df["review"] + " " +
            df["business"] + " " +
            df["location"]
        ).str.lower()

        query_clean = query.lower().strip()

        # ---------------------------------------------------------
        # Exact phrase search
        # ---------------------------------------------------------
        exact_mask = df["_search_text"].str.contains(
            query_clean,
            regex=False,
            na=False
        )

        matching = df[exact_mask].copy()

        print(
            f"[GoogleReviewsConnector] Exact matches: "
            f"{len(matching)}"
        )

        # ---------------------------------------------------------
        # Token-based fallback search
        # ---------------------------------------------------------
        if len(matching) < max_items:

            raw_tokens = re.findall(
                r"\b[a-zA-Z0-9]+\b",
                query_clean
            )

            stopwords = {
                "a", "an", "the", "and", "or",
                "for", "is", "of", "in", "on",
                "to", "with", "about", "at",
                "by", "from", "it", "this",
                "that"
            }

            tokens = [
                token
                for token in raw_tokens
                if token not in stopwords and len(token) > 1
            ]

            if not tokens:
                tokens = raw_tokens

            fallback_rows = []

            for token in tokens:

                token_mask = df["_search_text"].str.contains(
                    token,
                    regex=False,
                    na=False
                )

                token_matches = df[token_mask]

                for index, row in token_matches.iterrows():

                    if index not in matching.index:
                        fallback_rows.append(row)

                    if (
                        len(matching) + len(fallback_rows)
                        >= max_items
                    ):
                        break

                if (
                    len(matching) + len(fallback_rows)
                    >= max_items
                ):
                    break

            if fallback_rows:
                fallback_df = pd.DataFrame(fallback_rows)

                matching = pd.concat(
                    [matching, fallback_df],
                    ignore_index=False
                )

                matching = matching[
                    ~matching.index.duplicated(keep="first")
                ]

        # ---------------------------------------------------------
        # Limit results
        # ---------------------------------------------------------
        matching = matching.head(max_items)

        print(
            f"[GoogleReviewsConnector] Relevant records retrieved: "
            f"{len(matching)}"
        )

        # ---------------------------------------------------------
        # Standardize records
        # ---------------------------------------------------------
        results = []

        for _, row in matching.iterrows():

            # ---------------------------------------------
            # Convert rating to sentiment
            # ---------------------------------------------
            try:
                rating = float(row["rating"])

                if rating >= 4:
                    sentiment = "positive"
                elif rating <= 2:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

            except (ValueError, TypeError):
                rating = ""
                sentiment = "neutral"

            business = str(row.get("business", "") or "")
            location = str(row.get("location", "") or "")
            review = str(row.get("review", "") or "")
            date = str(row.get("date", "") or "")

            results.append({
                "source": "google",
                "title": business,
                "text": review,
                "url": "",
                "timestamp": date,
                "rating": str(rating),
                "category": sentiment,
                "product_name": business,
                "asin": "",
                "location": location
            })

        print(
            f"[GoogleReviewsConnector] Collection complete. "
            f"Relevant records retrieved: {len(results)}"
        )

        return results