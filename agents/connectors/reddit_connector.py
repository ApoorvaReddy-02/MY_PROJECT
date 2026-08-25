"""
reddit_connector.py
-------------------
Reddit Dataset Connector for the Data Collection Agent.

Uses the local Reddit_Data.csv dataset.

Supports:
- Strict keyword matching
- Product/category aliases
- Fallback searching
- Sentiment conversion
- Duplicate prevention

No generated/fake comments are used.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional

from .base_connector import BaseConnector


class RedditConnector(BaseConnector):
    """Reddit local dataset connector."""

    def __init__(self, dataset_path: Optional[str] = None):

        self.dataset_path = dataset_path or os.path.join(
            "datasets",
            "Reddit_Data.csv"
        )

        super().__init__(
            source_name="reddit",
            api_key=None
        )

        # Product aliases.
        self.aliases = {
            "hp laptop": [
                "hp laptop",
                "hp",
                "laptop",
                "hewlett",
                "notebook",
                "computer"
            ],

            "headphones": [
                "headphones",
                "headphone",
                "head set",
                "headset"
            ],

            "earphones": [
                "earphones",
                "earphone",
                "earbuds",
                "earbud",
                "headphones",
                "headphone"
            ],

            "usb cable": [
                "usb cable",
                "usb",
                "cable",
                "type-c",
                "type c",
                "charging cable",
                "charger"
            ],

            "mobile phone": [
                "mobile phone",
                "mobile",
                "phone",
                "smartphone",
                "cell phone",
                "iphone",
                "android"
            ],

            "laptop": [
                "laptop",
                "notebook",
                "computer"
            ],

            "iphone": [
                "iphone",
                "ios",
                "apple phone"
            ],

            "samsung": [
                "samsung",
                "galaxy"
            ]
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for reliable matching."""

        text = str(text or "").lower()

        # Normalize hyphens.
        text = text.replace("-", " ")

        # Remove punctuation.
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Normalize whitespace.
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _get_search_terms(self, query: str) -> List[str]:
        """
        Return search terms for the query.

        Exact query is preferred first, followed by aliases.
        """

        query_clean = self._normalize_text(query)

        terms = [query_clean]

        if query_clean in self.aliases:
            terms.extend(self.aliases[query_clean])

        # Remove duplicates while preserving order.
        unique_terms = []

        for term in terms:
            term = self._normalize_text(term)

            if term and term not in unique_terms:
                unique_terms.append(term)

        return unique_terms

    def _contains_term(
        self,
        text: str,
        term: str
    ) -> bool:
        """Check whether a term occurs in normalized text."""

        text = self._normalize_text(text)
        term = self._normalize_text(term)

        if not text or not term:
            return False

        return term in text

    def _get_sentiment(self, value: Any) -> str:
        """Convert Reddit category value into sentiment."""

        try:
            category = int(float(value))

            if category == 1:
                return "positive"

            if category == -1:
                return "negative"

            return "neutral"

        except (ValueError, TypeError):
            return "neutral"

    def fetch_data(
        self,
        query: str = "",
        max_items: int = 10
    ) -> List[Dict[str, Any]]:

        query = str(query or "").strip()

        if not query:
            return []

        if max_items <= 0:
            return []

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                "[RedditConnector] Dataset not found: "
                f"{self.dataset_path}"
            )

        print(
            "\n[RedditConnector] Searching local Reddit dataset..."
        )

        print(f"Query: {query}")

        # Load dataset.
        df = pd.read_csv(self.dataset_path)

        if "clean_comment" not in df.columns:
            raise ValueError(
                "[RedditConnector] Dataset must contain "
                "'clean_comment' column."
            )

        if "category" not in df.columns:
            raise ValueError(
                "[RedditConnector] Dataset must contain "
                "'category' column."
            )

        # Remove empty comments.
        df = df.dropna(
            subset=["clean_comment"]
        ).copy()

        df["clean_comment"] = (
            df["clean_comment"]
            .fillna("")
            .astype(str)
        )

        # Normalized search column.
        df["_normalized_comment"] = (
            df["clean_comment"]
            .apply(self._normalize_text)
        )

        query_normalized = self._normalize_text(query)

        # --------------------------------------------------
        # STEP 1: STRICT MATCH
        # --------------------------------------------------

        strict_mask = (
            df["_normalized_comment"]
            .str.contains(
                query_normalized,
                regex=False,
                na=False
            )
        )

        strict_matches = df[strict_mask]

        print(
            "[RedditConnector] Strict matches: "
            f"{len(strict_matches)}"
        )

        # --------------------------------------------------
        # STEP 2: FALLBACK MATCH
        # --------------------------------------------------

        matching = strict_matches.copy()

        search_terms = self._get_search_terms(query)

        if len(matching) < max_items:

            print(
                "[RedditConnector] Strict search returned "
                "fewer than requested."
            )

            print(
                "[RedditConnector] Fallback terms: "
                f"{search_terms}"
            )

            # Search each alias.
            for term in search_terms:

                if len(matching) >= max_items:
                    break

                print(
                    f"[RedditConnector] Fallback search: {term}"
                )

                mask = (
                    df["_normalized_comment"]
                    .str.contains(
                        term,
                        regex=False,
                        na=False
                    )
                )

                term_matches = df[mask]

                if term_matches.empty:
                    continue

                # Add only records not already selected.
                if not matching.empty:

                    new_matches = term_matches[
                        ~term_matches.index.isin(
                            matching.index
                        )
                    ]

                else:
                    new_matches = term_matches

                if not new_matches.empty:

                    matching = pd.concat(
                        [matching, new_matches]
                    )

                if len(matching) >= max_items:
                    break

        # --------------------------------------------------
        # LIMIT RESULTS
        # --------------------------------------------------

        matching = matching.head(max_items)

        # --------------------------------------------------
        # BUILD STANDARDIZED RECORDS
        # --------------------------------------------------

        results = []

        for index, row in matching.iterrows():

            text = str(
                row.get("clean_comment", "")
            ).strip()

            if not text:
                continue

            sentiment = self._get_sentiment(
                row.get("category", "")
            )

            results.append({

                "source": "reddit",

                "title": "Reddit Comment",

                "text": text,

                "url": "",

                "timestamp": "",

                "rating": "",

                "category": sentiment,

                "product_name": "",

                "asin": "",

                # IMPORTANT:
                # Tell DataCollectionAgent that this
                # record was obtained using a fallback
                # search term.
                "relevance_query": query,

                "relevance_terms": search_terms

            })

        print(
            "[RedditConnector] Collection complete. "
            f"Relevant records retrieved: {len(results)}"
        )

        return results