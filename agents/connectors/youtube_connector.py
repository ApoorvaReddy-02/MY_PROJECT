"""
youtube_connector.py
--------------------
YouTube Data Source Connector module for the Data Collection Agent.

Architecture & Security:
-----------------------
- Source Name: 'youtube'
- Target Service: YouTube Data API v3 (google-api-python-client)
- API Key handling: Loaded strictly from environment variable 'YOUTUBE_API_KEY' (via python-dotenv).
- Enforces query relevance checks on retrieved video snippets.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from googleapiclient.discovery import build
from .base_connector import BaseConnector

load_dotenv()


class YouTubeConnector(BaseConnector):
    """YouTube Data Source Connector performing authentic YouTube Data API v3 queries."""

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        super().__init__(source_name="youtube", api_key=api_key)

    def fetch_data(self, query: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches public YouTube video search results matching the query via YouTube Data API v3.
        """
        query = str(query or "").strip()
        if not query:
            return []

        if not self.api_key:
            raise ValueError(
                "[YouTubeConnector] Missing API Credentials: Set the 'YOUTUBE_API_KEY' "
                "environment variable in .env to enable live YouTube collection."
            )

        try:
            youtube = build("youtube", "v3", developerKey=self.api_key, cache_discovery=False)

            search_response = youtube.search().list(
                q=query,
                part="snippet",
                maxResults=max(10, min(max_items * 2, 50)),
                type="video"
            ).execute()

            items = search_response.get("items", [])
            results = []

            # Extract tokens for relevance check
            query_clean = query.lower()
            raw_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", query_clean)
            stopwords = {"a", "an", "the", "and", "or", "for", "is", "of", "in", "on", "to", "with", "about"}
            tokens = [t for t in raw_tokens if t not in stopwords and len(t) > 1]
            if not tokens:
                tokens = raw_tokens

            for item in items:
                snippet = item.get("snippet", {})
                video_id = item.get("id", {}).get("videoId", "")
                title = snippet.get("title", "")
                description = snippet.get("description", "")
                published_at = snippet.get("publishedAt", "")

                combined = f"{title} {description}".lower()
                if tokens:
                    if not all(token in combined for token in tokens):
                        continue

                url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""

                results.append({
                    "source": "youtube",
                    "text": description if description else title,
                    "title": title,
                    "url": url,
                    "timestamp": published_at
                })

                if len(results) >= max_items:
                    break

            return results

        except Exception as err:
            err_msg = str(err)
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[REDACTED_API_KEY]")
            raise RuntimeError(f"[YouTubeConnector] API Query failed: {err_msg}") from None
