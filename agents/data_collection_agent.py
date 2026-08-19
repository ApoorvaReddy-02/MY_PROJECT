"""
data_collection_agent.py
------------------------
Single Modular Data Collection Agent for the Agentic AI Sentiment System.

Architecture Overview (College Project Review):
----------------------------------------------
Instead of creating separate, monolithic agent classes for every website
(e.g., YouTubeAgent, RedditAgent, NewsAgent), this system utilizes a single,
decoupled DataCollectionAgent that orchestrates multiple modular connectors.

Connectors (YouTube, Reddit, News, Amazon, X, Flipkart, Bluesky) implement a shared
`BaseConnector` interface. The DataCollectionAgent dynamically dispatches requests
to the target connector, enforces query relevance filtering, and standardizes data output.
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


def is_record_relevant(item: Dict[str, Any], query: str) -> bool:
    """
    Verifies that the collected record contains the search query keywords.
    Prevents unrelated/fallback records from being displayed.
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

    # Check exact phrase match first
    if query_clean in combined_text:
        return True

    # Tokenize query into significant terms
    raw_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", query_clean)
    stopwords = {
        "a", "an", "the", "and", "or", "for", "is", "of", "in", "on", "to",
        "with", "about", "at", "by", "from", "it", "this", "that"
    }
    tokens = [t for t in raw_tokens if t not in stopwords and len(t) > 1]
    if not tokens:
        tokens = raw_tokens

    if not tokens:
        return True

    # All significant tokens must appear in the combined record content
    return all(token in combined_text for token in tokens)


class DataCollectionAgent:
    """
    Central Agent responsible for orchestrating multi-source data collection.

    Maintains a registry of data connectors and standardizes all collected outputs.
    """

    def __init__(self):
        self._connectors: Dict[str, BaseConnector] = {}

        # Register default source connector modules
        self.register_connector("youtube", YouTubeConnector())
        self.register_connector("reddit", RedditConnector())
        self.register_connector("news", NewsConnector())
        self.register_connector("amazon", AmazonConnector())
        self.register_connector("x", XConnector())
        self.register_connector("flipkart", FlipkartConnector())
        self.register_connector("bluesky", BlueskyConnector())

    def register_connector(self, source_name: str, connector: BaseConnector):
        key = source_name.lower().strip()
        self._connectors[key] = connector
        print(f"[DataCollectionAgent] Registered connector for source: '{key}'")

    def get_supported_sources(self) -> List[str]:
        return list(self._connectors.keys())

    def collect_data(
        self,
        query: str,
        source: str = "youtube",
        max_items: int = 10,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        source_key = source.lower().strip()
        if source_key not in self._connectors:
            raise ValueError(
                f"Unsupported source '{source}'. Currently registered sources: {self.get_supported_sources()}"
            )

        connector = self._connectors[source_key]

        if api_key:
            connector.api_key = api_key

        print(f"\n[DataCollectionAgent] Initiating collection:")
        print(f"  Source   : {source_key}")
        print(f"  Query    : '{query}'")
        print(f"  Max Items: {max_items}")

        raw_items = connector.fetch_data(query=query, max_items=max_items)

        standardized_items = []
        for item in raw_items:
            # Enforce relevance check
            if not is_record_relevant(item, query):
                continue

            standardized_items.append({
                "source": source_key,
                "text": str(item.get("text", "")),
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "timestamp": str(item.get("timestamp", "")),
                "rating": str(item.get("rating", "")),
                "category": str(item.get("category", "")),
                "product_name": str(item.get("product_name", "")),
                "asin": str(item.get("asin", ""))
            })

            if len(standardized_items) >= max_items:
                break

        print(f"[DataCollectionAgent] Collection complete. Standardized records retrieved: {len(standardized_items)}")
        return standardized_items


if __name__ == "__main__":
    agent = DataCollectionAgent()
    print("\nData Collection Agent Supported Sources:", agent.get_supported_sources())
