"""
youtube_connector.py
--------------------
YouTube Data Source Connector module for the Data Collection Agent.

Architecture & Security:
-----------------------
- Source Name: 'youtube'
- Target Service: YouTube Data API v3 (google-api-python-client)
- API Key handling: Loaded strictly from environment variable 'YOUTUBE_API_KEY' (via python-dotenv).
- No API keys are hard-coded or exposed in logs.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from googleapiclient.discovery import build
from .base_connector import BaseConnector

# Load environment variables securely from .env file
load_dotenv()


class YouTubeConnector(BaseConnector):
    """YouTube Data Source Connector performing authentic YouTube Data API v3 queries."""

    def __init__(self, api_key: Optional[str] = None):
        # Reads API key strictly from parameter or environment variable YOUTUBE_API_KEY
        api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        super().__init__(source_name="youtube", api_key=api_key)

    def fetch_data(self, query: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches public YouTube video search results matching the query via YouTube Data API v3.

        Args:
            query (str): Search topic or query (e.g. 'iPhone 17').
            max_items (int): Maximum records to retrieve.

        Returns:
            List[Dict[str, Any]]: Standardized records containing:
                - 'source': 'youtube'
                - 'text': video description / content snippet
                - 'title': video title
                - 'url': YouTube video watch link
                - 'timestamp': publication ISO timestamp
        """
        # Ensure API key is configured
        if not self.api_key:
            raise ValueError(
                "[YouTubeConnector] Missing API Credentials: Set the 'YOUTUBE_API_KEY' "
                "environment variable in .env to enable live YouTube collection."
            )

        try:
            # Initialize official YouTube Data API v3 client
            youtube = build("youtube", "v3", developerKey=self.api_key, cache_discovery=False)

            # Search public videos matching the query
            search_response = youtube.search().list(
                q=query,
                part="snippet",
                maxResults=max_items,
                type="video"
            ).execute()

            items = search_response.get("items", [])
            results = []

            for item in items:
                snippet = item.get("snippet", {})
                video_id = item.get("id", {}).get("videoId", "")
                title = snippet.get("title", "")
                description = snippet.get("description", "")
                published_at = snippet.get("publishedAt", "")

                url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""

                results.append({
                    "source": "youtube",
                    "text": description if description else title,
                    "title": title,
                    "url": url,
                    "timestamp": published_at
                })

            return results

        except Exception as err:
            # Mask API key if present in error string
            err_msg = str(err)
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[REDACTED_API_KEY]")
            raise RuntimeError(f"[YouTubeConnector] API Query failed: {err_msg}") from None
