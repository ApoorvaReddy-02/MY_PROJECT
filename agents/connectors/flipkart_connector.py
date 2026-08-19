import os
import requests
from typing import List, Dict, Any, Optional

from .base_connector import BaseConnector


class FlipkartConnector(BaseConnector):
    """
    Flipkart Product Reviews API connector using Parse API.
    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("PARSE_API_KEY")

        super().__init__(
            source_name="flipkart",
            api_key=api_key
        )

        self.base_url = "https://api.parse.bot"
        self.scraper_id = "dfeb72c1-9b76-4102-a752-70e10f3a0c50"

    def _call(
        self,
        endpoint: str,
        **params
    ) -> Dict[str, Any]:

        if not self.api_key:
            raise ValueError(
                "[FlipkartConnector] Missing API Credentials: "
                "Set PARSE_API_KEY environment variable."
            )

        url = (
            f"{self.base_url}/scraper/"
            f"{self.scraper_id}/{endpoint}"
        )

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    def fetch_data(
        self,
        query: str,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:

        """
        Query must be a Flipkart product review URL path.

        Example:
        /nivea-body-milk-nourishing-lotion-400ml-120-ml-pack-2/
        product-reviews/itmf6y86zahhzntz?pid=...
        """

        if not query:
            raise ValueError(
                "[FlipkartConnector] Product URL is required."
            )

        print("\n[FlipkartConnector] Fetching Flipkart reviews...")
        print(f"Product URL: {query}")

        result = self._call(
            "get_reviews",
            product_url=query,
            page="1",
            sort_order="MOST_RECENT",
            certified_buyer="false"
        )

        if result.get("status") != "success":
            print("[FlipkartConnector] API returned unsuccessful response.")
            return []

        data = result.get("data", {})

        reviews = data.get("reviews", [])

        standardized = []

        for review in reviews[:max_items]:

            rating = review.get("rating", "")

            if rating:
                if rating >= 4:
                    category = "positive"
                elif rating == 3:
                    category = "neutral"
                else:
                    category = "negative"
            else:
                category = ""

            standardized.append({
                "source": "flipkart",
                "text": review.get("text", ""),
                "title": review.get("title", "Flipkart Review"),
                "url": query,
                "timestamp": review.get("created", ""),
                "rating": rating,
                "category": category,
                "product_name": ""
            })

        print(
            f"[FlipkartConnector] API collection complete. "
            f"Records retrieved: {len(standardized)}"
        )

        return standardized