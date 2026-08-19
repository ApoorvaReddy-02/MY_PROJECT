"""
news_connector.py
-----------------
News Data Source Connector for the Data Collection Agent.

Uses NewsAPI to collect news articles matching a search query.
API key is loaded securely from the NEWS_API_KEY environment variable.
Enforces relevance filtering so only query-matching articles are returned.
"""

import os
import re
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

from .base_connector import BaseConnector

# Load variables from .env
load_dotenv()


class NewsConnector(BaseConnector):
    """News Data Source Connector using NewsAPI."""

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("NEWS_API_KEY")
        super().__init__(source_name="news", api_key=api_key)

    def fetch_data(
        self,
        query: str,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:

        query = str(query or "").strip()
        if not query:
            return []

        if not self.api_key:
            raise ValueError(
                "[NewsConnector] Missing API Credentials: "
                "Set NEWS_API_KEY in the .env file."
            )

        endpoint = "https://newsapi.org/v2/everything"

        params = {
            "q": query,
            "pageSize": max(10, min(max_items * 2, 100)),
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.api_key
        }

        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=15
            )

            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                raise RuntimeError(
                    f"NewsAPI returned an error: "
                    f"{data.get('message', 'Unknown error')}"
                )

            articles = data.get("articles", [])

            # Extract query tokens for relevance validation
            query_clean = query.lower()
            raw_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", query_clean)
            stopwords = {"a", "an", "the", "and", "or", "for", "is", "of", "in", "on", "to", "with", "about"}
            tokens = [t for t in raw_tokens if t not in stopwords and len(t) > 1]
            if not tokens:
                tokens = raw_tokens

            results = []

            for article in articles:
                title = article.get("title") or ""
                description = article.get("description") or ""
                content = article.get("content") or ""

                combined = f"{title} {description} {content}".lower()

                # Check token relevance
                if tokens:
                    if not all(token in combined for token in tokens):
                        continue

                text = description or content or title

                results.append({
                    "source": "news",
                    "text": text,
                    "title": title,
                    "url": article.get("url") or "",
                    "timestamp": article.get("publishedAt") or ""
                })

                if len(results) >= max_items:
                    break

            return results

        except requests.RequestException as err:
            raise RuntimeError(
                f"[NewsConnector] API request failed: {err}"
            ) from None

        except Exception as err:
            err_msg = str(err)
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[REDACTED_API_KEY]")

            raise RuntimeError(
                f"[NewsConnector] API Query failed: {err_msg}"
            ) from None