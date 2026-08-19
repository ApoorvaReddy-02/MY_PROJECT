"""
bluesky_connector.py
--------------------
Bluesky Data Source Connector for the Data Collection Agent.

Uses Bluesky's public AppView API to collect public posts.
Enforces query token relevance filtering.
"""

import re
import requests
from typing import List, Dict, Any
from .base_connector import BaseConnector


class BlueskyConnector(BaseConnector):
    """Bluesky public data connector."""

    def __init__(self):
        super().__init__(
            source_name="bluesky",
            api_key=None
        )

        self.base_url = "https://public.api.bsky.app/xrpc"

    def fetch_data(
        self,
        query: str,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:

        query = str(query or "").strip()
        if not query:
            return []

        try:
            response = requests.get(
                f"{self.base_url}/app.bsky.feed.searchPosts",
                params={
                    "q": query,
                    "limit": max(10, min(max_items * 2, 50))
                },
                timeout=15
            )

            response.raise_for_status()
            data = response.json()
            posts = data.get("posts", [])

            # Extract tokens for relevance check
            query_clean = query.lower()
            raw_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", query_clean)
            stopwords = {"a", "an", "the", "and", "or", "for", "is", "of", "in", "on", "to", "with", "about"}
            tokens = [t for t in raw_tokens if t not in stopwords and len(t) > 1]
            if not tokens:
                tokens = raw_tokens

            results = []

            for post in posts:
                record = post.get("record", {})
                text = record.get("text", "")
                created_at = record.get("createdAt", "")
                author = post.get("author", {})
                handle = author.get("handle", "")
                uri = post.get("uri", "")

                text_lower = text.lower()
                if tokens:
                    if not all(token in text_lower for token in tokens):
                        continue

                post_url = ""
                if handle and uri:
                    rkey = uri.split("/")[-1]
                    post_url = f"https://bsky.app/profile/{handle}/post/{rkey}"

                results.append({
                    "source": "bluesky",
                    "text": text,
                    "title": text[:100],
                    "url": post_url,
                    "timestamp": created_at,
                    "author": handle
                })

                if len(results) >= max_items:
                    break

            return results

        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"[BlueskyConnector] Failed to fetch data: {e}"
            ) from e