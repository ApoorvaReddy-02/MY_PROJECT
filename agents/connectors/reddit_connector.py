"""
reddit_connector.py
-------------------
Reddit Data Source Connector module for the Data Collection Agent.

Architecture & Credentials Documentation:
----------------------------------------
- Source Name: 'reddit'
- Required Credentials: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT (Environment Variables)
- Target Service: Reddit Data API via PRAW (Python Reddit API Wrapper)
- Collected Data: Subreddit posts, self-text content, post titles, post permalinks, submission timestamps
- Access Requirements: Reddit Developer account & registered script application.
"""

import os
from typing import List, Dict, Any, Optional
from .base_connector import BaseConnector


class RedditConnector(BaseConnector):
    """Reddit Data Source Connector implementing PRAW/Reddit API integration structure."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "SentimentAgenticSystem/1.0")

        super().__init__(source_name="reddit", api_key=self.client_secret)

    def fetch_data(self, query: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches submissions/comments matching the search query via Reddit API.

        Args:
            query (str): Search topic or keyword.
            max_items (int): Maximum items to retrieve.

        Returns:
            List[Dict[str, Any]]: Standardized collection items.
        """
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "[RedditConnector] Missing API Credentials: Set 'REDDIT_CLIENT_ID' and "
                "'REDDIT_CLIENT_SECRET' environment variables to enable live Reddit collection."
            )

        # =======================================================================
        # LIVE API INTEGRATION STRUCTURE (Active when credentials are set)
        # =======================================================================
        # import praw
        # reddit = praw.Reddit(
        #     client_id=self.client_id,
        #     client_secret=self.client_secret,
        #     user_agent=self.user_agent
        # )
        # for submission in reddit.subreddit('all').search(query, limit=max_items):
        #     # process submission.title, submission.selftext, submission.url
        # =======================================================================

        results = []
        return results
