"""
data_collection_agent.py
------------------------
Single Modular Data Collection Agent for the Agentic AI Sentiment System.

Architecture Overview (College Project Review):
----------------------------------------------
Instead of creating separate, monolithic agent classes for every website
(e.g., YouTubeAgent, RedditAgent, NewsAgent), this system utilizes a single,
decoupled DataCollectionAgent that orchestrates multiple modular connectors.

Connectors (YouTube, Reddit, News, Amazon, X) implement a shared `BaseConnector`
interface. The DataCollectionAgent dynamically dispatches requests to the target connector
and enforces a unified data schema across all sources.
"""

from typing import List, Dict, Any, Optional
from .connectors.base_connector import BaseConnector
from .connectors.youtube_connector import YouTubeConnector
from .connectors.reddit_connector import RedditConnector
from .connectors.news_connector import NewsConnector
from .connectors.amazon_connector import AmazonConnector
from .connectors.x_connector import XConnector


class DataCollectionAgent:
    """
    Central Agent responsible for orchestrating multi-source data collection.

    Maintains a registry of data connectors and standardizes all collected outputs.
    """

    def __init__(self):
        # Dictionary mapping source names to BaseConnector instances
        self._connectors: Dict[str, BaseConnector] = {}

        # Register default source connector modules
        self.register_connector("youtube", YouTubeConnector())
        self.register_connector("reddit", RedditConnector())
        self.register_connector("news", NewsConnector())
        self.register_connector("amazon", AmazonConnector())
        self.register_connector("x", XConnector())

    def register_connector(self, source_name: str, connector: BaseConnector):
        """
        Registers a new data source connector with the agent.

        Args:
            source_name (str): Identifier for the source (e.g., 'youtube', 'reddit', 'news', 'amazon', 'x')
            connector (BaseConnector): Instance of a class inheriting from BaseConnector
        """
        key = source_name.lower().strip()
        self._connectors[key] = connector
        print(f"[DataCollectionAgent] Registered connector for source: '{key}'")

    def get_supported_sources(self) -> List[str]:
        """Returns a list of currently registered source identifiers."""
        return list(self._connectors.keys())

    def collect_data(
        self,
        query: str,
        source: str = "youtube",
        max_items: int = 10,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Collects data for a search query from the specified source.

        Args:
            query (str): Search topic or keyword.
            source (str): Target data source name (e.g., 'youtube', 'reddit', 'news', 'amazon', 'x').
            max_items (int): Maximum items to collect.
            api_key (Optional[str]): Source-specific API key if available.

        Returns:
            List[Dict[str, Any]]: Standardized records formatted with:
                - 'source': str
                - 'text': str
                - 'title': str
                - 'url': str
                - 'timestamp': str
        """
        source_key = source.lower().strip()
        if source_key not in self._connectors:
            raise ValueError(
                f"Unsupported source '{source}'. Currently registered sources: {self.get_supported_sources()}"
            )

        connector = self._connectors[source_key]

        # Pass API key if provided at runtime
        if api_key:
            connector.api_key = api_key

        print(f"\n[DataCollectionAgent] Initiating collection:")
        print(f"  Source   : {source_key}")
        print(f"  Query    : '{query}'")
        print(f"  Max Items: {max_items}")

        # Delegate data fetching to the specific connector instance
        raw_items = connector.fetch_data(query=query, max_items=max_items)

        # Enforce Common Output Standard across all data sources
        standardized_items = []
        for item in raw_items:
            standardized_items.append({
                "source": source_key,
                "text": str(item.get("text", "")),
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "timestamp": str(item.get("timestamp", ""))
            })

        print(f"[DataCollectionAgent] Collection complete. Standardized records retrieved: {len(standardized_items)}")
        return standardized_items


if __name__ == "__main__":
    agent = DataCollectionAgent()
    print("\nData Collection Agent Supported Sources:", agent.get_supported_sources())
