"""
base_connector.py
------------------
Abstract Base Class (Interface) for all data-source connectors.

Design Pattern: Strategy Pattern / Modular Connector Architecture
------------------------------------------------------------------
Every data source (YouTube, Reddit, News, Product Reviews) must inherit
from BaseConnector and implement the fetch_data() method.

This ensures a uniform API across all sources and allows the DataCollectionAgent
to interact with any data source seamlessly without changing its core logic.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseConnector(ABC):
    """Abstract interface that all data connectors must implement."""

    def __init__(self, source_name: str, api_key: Optional[str] = None):
        self.source_name = source_name
        self.api_key = api_key

    @abstractmethod
    def fetch_data(self, query: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches data from the target data source for a given search query.

        Args:
            query (str): The search query or topic to collect data for.
            max_items (int): Maximum number of items to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of items, where each item is formatted with:
                - 'source': str
                - 'text': str
                - 'title': str
                - 'url': str
                - 'timestamp': str
        """
        pass
