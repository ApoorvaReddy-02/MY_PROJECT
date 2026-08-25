"""
data_collection_agent.py
------------------------
Single Modular Data Collection Agent for the Agentic AI Sentiment System.

Architecture Overview (College Project Review):
----------------------------------------------
Instead of creating separate, monolithic agent classes for every website
(e.g., YouTubeAgent, RedditAgent, NewsAgent), this system utilizes a single,
decoupled DataCollectionAgent that orchestrates multiple modular connectors.

Connectors (YouTube, Reddit, News, Amazon, X, Flipkart, Bluesky) implement a
shared BaseConnector interface. The DataCollectionAgent dynamically dispatches
requests to the target connector, enforces query relevance filtering, and
standardizes data output.
"""

import re
from typing import List, Dict, Any, Optional

from .connectors.base_connector import BaseConnector
from .connectors.youtube_connector import YouTubeConnector
from .connectors.reddit_connector import RedditConnector
from .connectors.news_connector import NewsConnector
from .connectors.amazon_connector import AmazonConnector
from .connectors.x_connector import XConnector
from .connectors.flipkart_connector import FlipkartConnector
from .connectors.bluesky_connector import BlueskyConnector
from .connectors.google_reviews_connector import GoogleReviewsConnector


def is_record_relevant(
    item: Dict[str, Any],
    query: str
) -> bool:
    """
    Verifies that the collected record contains the search query keywords.

    Normal searches require all significant query tokens.

    Connector-specific fallback terms are handled separately inside
    DataCollectionAgent.collect_data().
    """

    if not query or not str(query).strip():
        return True

    query_clean = str(query).strip().lower()

    # Combined text fields from item
    combined_text = " ".join([
        str(item.get("title", "") or ""),
        str(item.get("text", "") or ""),
        str(item.get("product_name", "") or ""),
        str(item.get("category", "") or "")
    ]).lower()

    if not combined_text.strip():
        return False

    # Exact phrase match
    if query_clean in combined_text:
        return True

    # Tokenize query
    raw_tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query_clean
    )

    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "is",
        "of",
        "in",
        "on",
        "to",
        "with",
        "about",
        "at",
        "by",
        "from",
        "it",
        "this",
        "that"
    }

    tokens = [
        token
        for token in raw_tokens
        if token not in stopwords and len(token) > 1
    ]

    if not tokens:
        tokens = raw_tokens

    if not tokens:
        return True

    # Normal strict relevance check:
    # all significant query tokens must occur.
    return all(
        token in combined_text
        for token in tokens
    )


class DataCollectionAgent:
    """
    Central Agent responsible for orchestrating multi-source data collection.

    Maintains a registry of data connectors and standardizes all collected
    outputs.
    """

    def __init__(self):

        self._connectors: Dict[str, BaseConnector] = {}

        # Register default source connector modules
        self.register_connector(
            "youtube",
            YouTubeConnector()
        )

        self.register_connector(
            "reddit",
            RedditConnector()
        )

        self.register_connector(
            "news",
            NewsConnector()
        )

        self.register_connector(
            "amazon",
            AmazonConnector()
        )

        self.register_connector(
            "x",
            XConnector()
        )

        self.register_connector(
            "flipkart",
            FlipkartConnector()
        )

        self.register_connector(
            "bluesky",
            BlueskyConnector()
        )
        self.register_connector(
            "google", GoogleReviewsConnector())

    def register_connector(
        self,
        source_name: str,
        connector: BaseConnector
    ):
        """Register a connector under its source name."""

        key = source_name.lower().strip()

        self._connectors[key] = connector

        print(
            f"[DataCollectionAgent] Registered connector "
            f"for source: '{key}'"
        )

    def get_supported_sources(self) -> List[str]:
        """Return all registered data sources."""

        return list(
            self._connectors.keys()
        )

    def collect_data(
        self,
        query: str,
        source: str = "youtube",
        max_items: int = 10,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect data from the requested source.

        Connector-specific fallback terms are supported through
        the optional 'relevance_terms' field.
        """

        source_key = source.lower().strip()

        # Check source
        if source_key not in self._connectors:

            raise ValueError(
                f"Unsupported source '{source}'. "
                f"Currently registered sources: "
                f"{self.get_supported_sources()}"
            )

        connector = self._connectors[source_key]

        # Set API key when supplied
        if api_key:
            connector.api_key = api_key

        print(
            "\n[DataCollectionAgent] Initiating collection:"
        )

        print(
            f"  Source   : {source_key}"
        )

        print(
            f"  Query    : '{query}'"
        )

        print(
            f"  Max Items: {max_items}"
        )

        # --------------------------------------------------
        # COLLECT RAW DATA
        # --------------------------------------------------

        raw_items = connector.fetch_data(
            query=query,
            max_items=max_items
        )

        standardized_items = []

        # --------------------------------------------------
        # STANDARDIZE AND FILTER RESULTS
        # --------------------------------------------------

        for item in raw_items:

            # --------------------------------------------------
            # NORMAL RELEVANCE CHECK
            # --------------------------------------------------

            if not is_record_relevant(
                item,
                query
            ):

                # --------------------------------------------------
                # CONNECTOR-SPECIFIC FALLBACK
                # --------------------------------------------------

                relevance_terms = item.get(
                    "relevance_terms",
                    []
                )

                # No fallback terms means the record
                # is not relevant.
                if not relevance_terms:
                    continue

                combined_text = " ".join([
                    str(
                        item.get("title", "") or ""
                    ),
                    str(
                        item.get("text", "") or ""
                    ),
                    str(
                        item.get("product_name", "") or ""
                    ),
                    str(
                        item.get("category", "") or ""
                    )
                ]).lower()

                fallback_match = False

                # Check whether at least one approved
                # fallback term exists in the record.
                for term in relevance_terms:

                    term = str(
                        term
                    ).strip().lower()

                    if (
                        term
                        and term in combined_text
                    ):
                        fallback_match = True
                        break

                # Reject if no fallback term matches.
                if not fallback_match:
                    continue

            # --------------------------------------------------
            # STANDARDIZED OUTPUT
            # --------------------------------------------------

            standardized_items.append({

                "source": source_key,

                "text": str(
                    item.get("text", "")
                ),

                "title": str(
                    item.get("title", "")
                ),

                "url": str(
                    item.get("url", "")
                ),

                "timestamp": str(
                    item.get("timestamp", "")
                ),

                "rating": str(
                    item.get("rating", "")
                ),

                "category": str(
                    item.get("category", "")
                ),

                "product_name": str(
                    item.get("product_name", "")
                ),

                "asin": str(
                    item.get("asin", "")
                )
            })

            # Stop once requested number is reached.
            if len(standardized_items) >= max_items:
                break

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        print(
            "[DataCollectionAgent] Collection complete. "
            f"Standardized records retrieved: "
            f"{len(standardized_items)}"
        )

        return standardized_items


if __name__ == "__main__":

    agent = DataCollectionAgent()

    print(
        "\nData Collection Agent Supported Sources:",
        agent.get_supported_sources()
    )