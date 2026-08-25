import os
import requests
import pandas as pd

from typing import List, Dict, Any, Optional

from .base_connector import BaseConnector


class FlipkartConnector(BaseConnector):
    """
    Flipkart Data Collection Connector.

    Primary source:
        Parse API

    Fallback source:
        Local Dataset-SA.csv

    If the Parse API is unavailable, the connector
    automatically searches the local Flipkart dataset.
    """

    def __init__(self, api_key: Optional[str] = None):

        api_key = api_key or os.getenv("PARSE_API_KEY")

        super().__init__(
            source_name="flipkart",
            api_key=api_key
        )

        self.base_url = "https://api.parse.bot"

        self.scraper_id = "95abe413-7660-4346-b524-760385511510"

        self.dataset_path = os.path.join(
            os.getcwd(),
            "datasets",
            "Dataset-SA.csv"
        )

    # ============================================================
    # PARSE API CALL
    # ============================================================

    def _call(
        self,
        endpoint: str,
        **params
    ) -> Dict[str, Any]:

        if not self.api_key:
            raise ValueError(
                "[FlipkartConnector] Missing API Credentials."
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
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    # ============================================================
    # SEARCH PRODUCTS USING API
    # ============================================================

    def _search_products(
        self,
        query: str
    ) -> List[Dict[str, Any]]:

        print("\n[FlipkartConnector] Searching products...")
        print(f"Query: {query}")

        try:

            result = self._call(
                "search_products",
                query=query
            )

        except requests.RequestException as exc:

            print(
                "[FlipkartConnector] Product search failed: "
                f"{exc}"
            )

            return []

        if result.get("status") != "success":

            print(
                "[FlipkartConnector] Product search returned "
                "unsuccessful response."
            )

            return []

        data = result.get("data", {})

        products = data.get("products", [])

        print(
            f"[FlipkartConnector] Products found: "
            f"{len(products)}"
        )

        return products

    # ============================================================
    # GET REVIEWS USING API
    # ============================================================

    def _get_reviews(
        self,
        product_url: str
    ) -> List[Dict[str, Any]]:

        try:

            result = self._call(
                "get_reviews",
                product_url=product_url,
                page="1",
                sort_order="MOST_RECENT",
                certified_buyer="false"
            )

        except requests.RequestException as exc:

            print(
                "[FlipkartConnector] Review request failed: "
                f"{exc}"
            )

            return []

        if result.get("status") != "success":

            return []

        data = result.get("data", {})

        return data.get("reviews", [])

    # ============================================================
    # API DATA COLLECTION
    # ============================================================

    def _fetch_from_api(
        self,
        query: str,
        max_items: int
    ) -> List[Dict[str, Any]]:

        products = self._search_products(query)

        if not products:

            return []

        standardized = []

        for product in products:

            if len(standardized) >= max_items:

                break

            product_name = str(
                product.get("name", "")
            ).strip()

            product_url = str(
                product.get("url", "")
            ).strip()

            if not product_name or not product_url:

                continue

            reviews = self._get_reviews(product_url)

            if not reviews:

                continue

            if product_url.startswith("/"):

                original_url = (
                    "https://www.flipkart.com"
                    + product_url
                )

            else:

                original_url = product_url

            for review in reviews:

                if len(standardized) >= max_items:

                    break

                text = str(
                    review.get("text", "")
                ).strip()

                title = str(
                    review.get("title", "")
                ).strip()

                if not text and not title:

                    continue

                rating = str(
                    review.get("rating", "")
                ).strip()

                category = self._get_category(rating)

                standardized.append({
                    "source": "flipkart",
                    "text": text,
                    "title": title or "Flipkart Review",
                    "url": original_url,
                    "timestamp": str(
                        review.get("created", "")
                    ),
                    "rating": rating,
                    "category": category,
                    "product_name": product_name,
                    "product_id": str(
                        product.get("product_id", "")
                    ),
                    "review_count": str(
                        product.get("review_count", "")
                    )
                })

        return standardized[:max_items]

    # ============================================================
    # RATING → SENTIMENT
    # ============================================================

    @staticmethod
    def _get_category(
        rating: str
    ) -> str:

        if rating == "":

            return ""

        try:

            numeric_rating = float(rating)

            if numeric_rating >= 4:

                return "positive"

            elif numeric_rating == 3:

                return "neutral"

            elif numeric_rating <= 2:

                return "negative"

        except (ValueError, TypeError):

            return ""

        return ""

    # ============================================================
    # LOCAL DATASET SEARCH
    # ============================================================

    def _fetch_from_dataset(
        self,
        query: str,
        max_items: int
    ) -> List[Dict[str, Any]]:

        print(
            "\n[FlipkartConnector] Searching local "
            "Flipkart dataset..."
        )

        print(f"Query: {query}")

        if not os.path.exists(self.dataset_path):

            print(
                "[FlipkartConnector] Dataset not found:"
            )

            print(self.dataset_path)

            return []

        try:

            print(
                "[FlipkartConnector] Loading Dataset-SA.csv..."
            )

            df = pd.read_csv(
                self.dataset_path,
                encoding="latin1",
                low_memory=False
            )

        except Exception as exc:

            print(
                "[FlipkartConnector] Dataset loading failed: "
                f"{exc}"
            )

            return []

        print(
            f"[FlipkartConnector] Dataset loaded. "
            f"Rows: {len(df)}"
        )

        query_tokens = [
            token.lower()
            for token in query.split()
            if token.strip()
        ]

        if not query_tokens:

            return []

        product_column = "product_name"
        review_column = "Review"
        summary_column = "Summary"
        rating_column = "Rate"

        if product_column not in df.columns:

            print(
                "[FlipkartConnector] product_name column "
                "not found."
            )

            return []

        df[product_column] = (
            df[product_column]
            .fillna("")
            .astype(str)
        )

        df[review_column] = (
            df[review_column]
            .fillna("")
            .astype(str)
        )

        df[summary_column] = (
            df[summary_column]
            .fillna("")
            .astype(str)
        )

        df[rating_column] = (
            df[rating_column]
            .fillna("")
            .astype(str)
        )

        # --------------------------------------------------------
        # Search product name + review + summary
        # --------------------------------------------------------

        combined_text = (
            df[product_column]
            + " "
            + df[review_column]
            + " "
            + df[summary_column]
        ).str.lower()

        mask = pd.Series(
            True,
            index=df.index
        )

        for token in query_tokens:

            mask = (
                mask
                & combined_text.str.contains(
                    token,
                    case=False,
                    na=False,
                    regex=False
                )
            )

        matches = df[mask]

        print(
            f"[FlipkartConnector] Dataset matches found: "
            f"{len(matches)}"
        )

        standardized = []

        for _, row in matches.iterrows():

            if len(standardized) >= max_items:

                break

            product_name = str(
                row.get(product_column, "")
            ).strip()

            review = str(
                row.get(review_column, "")
            ).strip()

            summary = str(
                row.get(summary_column, "")
            ).strip()

            rating = str(
                row.get(rating_column, "")
            ).strip()

            if not review and not summary:

                continue

            category = self._get_category(
                rating
            )

            standardized.append({
                "source": "flipkart",
                "text": review,
                "title": summary or "Flipkart Review",
                "url": "",
                "timestamp": "",
                "rating": rating,
                "category": category,
                "product_name": product_name,
                "product_id": "",
                "review_count": ""
            })

        print(
            "[FlipkartConnector] Dataset fallback complete. "
            f"Relevant records retrieved: "
            f"{len(standardized)}"
        )

        return standardized

    # ============================================================
    # MAIN FETCH FUNCTION
    # ============================================================

    def fetch_data(
        self,
        query: str,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():

            raise ValueError(
                "[FlipkartConnector] Search query is required."
            )

        query = query.strip()

        # --------------------------------------------------------
        # STEP 1: Try real Flipkart API
        # --------------------------------------------------------

        api_results = []

        if self.api_key:

            try:

                api_results = self._fetch_from_api(
                    query,
                    max_items
                )

            except Exception as exc:

                print(
                    "[FlipkartConnector] API unavailable: "
                    f"{exc}"
                )

        # --------------------------------------------------------
        # STEP 2: Use API results if available
        # --------------------------------------------------------

        if api_results:

            print(
                "[FlipkartConnector] API collection complete. "
                f"Real customer reviews retrieved: "
                f"{len(api_results)}"
            )

            return api_results[:max_items]

        # --------------------------------------------------------
        # STEP 3: Dataset fallback
        # --------------------------------------------------------

        print(
            "[FlipkartConnector] API unavailable or returned "
            "no reviews."
        )

        print(
            "[FlipkartConnector] Switching to Dataset-SA.csv..."
        )

        dataset_results = self._fetch_from_dataset(
            query,
            max_items
        )

        print(
            "[FlipkartConnector] Final records retrieved: "
            f"{len(dataset_results)}"
        )

        return dataset_results[:max_items]