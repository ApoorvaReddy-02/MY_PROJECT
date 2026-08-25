"""
amazon_connector.py
-------------------
Amazon Dataset Connector for the Data Collection Agent.

Uses ONLY local Amazon CSV datasets.

Datasets:
    1. datasets/amazon.csv
    2. datasets/amazon_electronics_review_sentiment.csv

No live Amazon API.
"""

import os
import re
from typing import List, Dict, Any, Optional

import pandas as pd

from .base_connector import BaseConnector


class AmazonConnector(BaseConnector):
    """Amazon review dataset connector with multi-dataset search."""

    def __init__(self, dataset_path: Optional[str] = None):

        super().__init__(
            source_name="amazon",
            api_key=None
        )

        # ---------------------------------------------------------
        # DATASET PATHS
        # ---------------------------------------------------------

        base_dataset_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "datasets"
            )
        )

        if dataset_path:
            self.dataset_paths = [dataset_path]
        else:
            self.dataset_paths = [
                os.path.join(
                    base_dataset_dir,
                    "amazon.csv"
                ),
                os.path.join(
                    base_dataset_dir,
                    "amazon_electronics_review_sentiment.csv"
                )
            ]

        print("[AmazonConnector] Dataset mode enabled.")

        for path in self.dataset_paths:
            print(f"[AmazonConnector] Dataset: {path}")

        # ---------------------------------------------------------
        # CACHE
        # ---------------------------------------------------------

        self._old_dataset = None
        self._electronics_dataset = None

    # =============================================================
    # LOAD ORIGINAL AMAZON DATASET
    # =============================================================

    def _load_old_dataset(self):

        if self._old_dataset is not None:
            return self._old_dataset

        path = self.dataset_paths[0]

        if not os.path.exists(path):
            print(
                f"[AmazonConnector] Original dataset not found: "
                f"{path}"
            )
            self._old_dataset = pd.DataFrame()
            return self._old_dataset

        print(
            f"[AmazonConnector] Loading original Amazon dataset..."
        )

        try:
            df = pd.read_csv(
                path,
                encoding="utf-8",
                low_memory=False
            )

        except UnicodeDecodeError:

            df = pd.read_csv(
                path,
                encoding="latin1",
                low_memory=False
            )

        except Exception as e:

            print(
                f"[AmazonConnector] Failed to load amazon.csv: {e}"
            )

            self._old_dataset = pd.DataFrame()
            return self._old_dataset

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        df = df.fillna("")

        print(
            f"[AmazonConnector] Original Amazon dataset loaded. "
            f"Rows: {len(df)}"
        )

        self._old_dataset = df

        return df

    # =============================================================
    # LOAD NEW ELECTRONICS DATASET
    # =============================================================

    def _load_electronics_dataset(self):

        if self._electronics_dataset is not None:
            return self._electronics_dataset

        if len(self.dataset_paths) < 2:
            self._electronics_dataset = pd.DataFrame()
            return self._electronics_dataset

        path = self.dataset_paths[1]

        if not os.path.exists(path):

            print(
                "[AmazonConnector] Electronics dataset not found: "
                f"{path}"
            )

            self._electronics_dataset = pd.DataFrame()

            return self._electronics_dataset

        print(
            "[AmazonConnector] Loading Amazon electronics "
            "review dataset..."
        )

        try:

            df = pd.read_csv(
                path,
                encoding="utf-8",
                low_memory=False
            )

        except UnicodeDecodeError:

            df = pd.read_csv(
                path,
                encoding="latin1",
                low_memory=False
            )

        except Exception as e:

            print(
                "[AmazonConnector] Failed to load "
                f"electronics dataset: {e}"
            )

            self._electronics_dataset = pd.DataFrame()

            return self._electronics_dataset

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        df = df.fillna("")

        print(
            "[AmazonConnector] Electronics dataset loaded. "
            f"Rows: {len(df)}"
        )

        self._electronics_dataset = df

        return df

    # =============================================================
    # QUERY TOKENIZATION
    # =============================================================

    def _get_query_information(self, query: str):

        query_clean = query.lower().strip()

        raw_tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query_clean
        )

        stopwords = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "for",
            "is",
            "of",
            "in",
            "on",
            "to",
            "with",
            "about",
            "at",
            "by",
            "from",
            "it",
            "this",
            "that"
        }

        tokens = [
            token
            for token in raw_tokens
            if token not in stopwords
            and len(token) > 1
        ]

        if not tokens:
            tokens = raw_tokens

        return query_clean, tokens

    # =============================================================
    # ORIGINAL AMAZON DATASET SEARCH
    # =============================================================

    def _search_old_dataset(
        self,
        df: pd.DataFrame,
        query: str,
        tokens: List[str],
        max_items: int
    ) -> List[Dict[str, Any]]:

        if df.empty:
            return []

        # ---------------------------------------------------------
        # PRODUCT TYPE MAP
        # ---------------------------------------------------------

        product_type_map = {

            "phone": {
                "smartphones",
                "basicmobiles",
                "mobiles",
                "cellphone",
                "smartphone"
            },

            "phones": {
                "smartphones",
                "basicmobiles",
                "mobiles",
                "cellphone",
                "smartphone"
            },

            "smartphone": {
                "smartphones",
                "basicmobiles",
                "mobiles",
                "cellphone"
            },

            "smartphones": {
                "smartphones",
                "basicmobiles",
                "mobiles",
                "cellphone"
            },

            "mobile": {
                "smartphones",
                "basicmobiles",
                "mobiles",
                "cellphone"
            },

            "iphone": {
                "smartphones",
                "basicmobiles",
                "mobiles",
                "cellphone",
                "iphone"
            },

            "laptop": {
                "laptops"
            },

            "laptops": {
                "laptops"
            },

            "notebook": {
                "laptops"
            },

            "headphones": {
                "headphones",
                "in-ear",
                "over-ear",
                "on-ear",
                "earbuds",
                "headsets"
            },

            "headphone": {
                "headphones",
                "in-ear",
                "over-ear",
                "on-ear",
                "earbuds",
                "headsets"
            },

            "earphones": {
                "headphones",
                "in-ear",
                "over-ear",
                "on-ear",
                "earbuds"
            },

            "speaker": {
                "bluetoothspeakers",
                "portablespeakers",
                "speakers",
                "soundbars"
            },

            "speakers": {
                "bluetoothspeakers",
                "portablespeakers",
                "speakers",
                "soundbars"
            },

            "cable": {
                "cables",
                "usbcables",
                "hdmicables"
            },

            "cables": {
                "cables",
                "usbcables",
                "hdmicables"
            },

            "mouse": {
                "mice",
                "mouse"
            },

            "mice": {
                "mice",
                "mouse"
            },

            "keyboard": {
                "keyboards"
            },

            "keyboards": {
                "keyboards"
            },

            "watch": {
                "smartwatches"
            },

            "smartwatch": {
                "smartwatches"
            },

            "tv": {
                "televisions",
                "smarttelevisions"
            },

            "television": {
                "televisions",
                "smarttelevisions"
            },

            "printer": {
                "printers"
            },

            "tablet": {
                "tablets"
            },

            "monitor": {
                "monitors"
            }
        }

        # ---------------------------------------------------------
        # ACCESSORIES
        # ---------------------------------------------------------

        accessory_nouns = {
            "cable",
            "charger",
            "adapter",
            "caddy",
            "converter",
            "connector",
            "case",
            "cover",
            "glass",
            "protector",
            "cord",
            "wire",
            "stand",
            "bag",
            "sleeve",
            "holder",
            "table",
            "mount",
            "strap",
            "mat",
            "pad",
            "power bank",
            "stylus",
            "skin",
            "screen guard",
            "battery",
            "cartridge",
            "ink",
            "printer",
            "webcam",
            "mouse",
            "keyboard",
            "lapdesk",
            "fan",
            "cooling pad",
            "hub",
            "dock",
            "remote",
            "transmitter",
            "receiver"
        }

        accessory_leaf_categories = {
            "cables",
            "usbcables",
            "hdmicables",
            "chargers",
            "wallchargers",
            "powerbanks",
            "lapdesks",
            "laptopaccessories",
            "mice",
            "keyboards",
            "remotecontrols",
            "microsd",
            "memorycards",
            "cases",
            "covers",
            "screenprotectors",
            "stylus",
            "graphicstablets",
            "networkadapters",
            "wirelessusbadapters",
            "printers",
            "cleaning",
            "mousepad",
            "mousepads",
            "stationery",
            "officeproducts",
            "paper",
            "adapters",
            "transmitters",
            "receivers",
            "accessories",
            "mobileaccessories"
        }

        # ---------------------------------------------------------
        # COMPATIBILITY PHRASES
        # ---------------------------------------------------------

        compat_delimiters = [
            "compatible for",
            "compatible with",
            "for iphone",
            "for samsung",
            "for laptop",
            "for pc",
            "for android",
            "for ipad",
            "case for",
            "cover for",
            "stand for",
            "cable for",
            "bag for",
            "sleeve for",
            "adapter for",
            "charger for",
            "protector for",
            "suitable for",
            "fits "
        ]

        # ---------------------------------------------------------
        # REQUESTED TYPES / BRANDS
        # ---------------------------------------------------------

        requested_types = set()
        brand_tokens = []
        modifiers = []

        for token in tokens:

            if token in product_type_map:

                requested_types.add(token)

                if token == "iphone":
                    brand_tokens.append("iphone")

            elif token in {
                "gaming",
                "wireless",
                "bluetooth",
                "smart",
                "portable",
                "fast",
                "hd",
                "4k",
                "foldable"
            }:

                modifiers.append(token)

            else:

                brand_tokens.append(token)

        main_device_requested = any(
            product_type in {
                "phone",
                "phones",
                "smartphone",
                "smartphones",
                "mobile",
                "iphone",
                "laptop",
                "laptops",
                "notebook",
                "tv",
                "television",
                "tablet"
            }
            for product_type in requested_types
        )

        user_asks_accessory = any(
            token in accessory_nouns
            for token in tokens
        )

        scored_records = []

        # ---------------------------------------------------------
        # SEARCH ROWS
        # ---------------------------------------------------------

        for _, row in df.iterrows():

            product_name = str(
                row.get("product_name", "")
            ).strip()

            category = str(
                row.get("category", "")
            ).strip()

            about_product = str(
                row.get("about_product", "")
            ).strip()

            review_title = str(
                row.get("review_title", "")
            ).strip()

            review_content = str(
                row.get("review_content", "")
            ).strip()

            rating = str(
                row.get("rating", "")
            ).strip()

            product_link = str(
                row.get("product_link", "")
            ).strip()

            product_id = str(
                row.get("product_id", "")
            ).strip()

            p_lower = product_name.lower()
            c_lower = category.lower()

            # -----------------------------------------------------
            # Remove compatibility section
            # -----------------------------------------------------

            main_p = p_lower

            for delimiter in compat_delimiters:

                if delimiter in p_lower:

                    main_p = p_lower[
                        :p_lower.find(delimiter)
                    ]

                    break

            # -----------------------------------------------------
            # Category leaf
            # -----------------------------------------------------

            cat_leaves = [
                part.strip()
                .lower()
                .replace("&", "")
                .replace(" ", "")
                for part in c_lower.split("|")
            ]

            leaf = (
                cat_leaves[-1]
                if cat_leaves
                else c_lower
            )

            # -----------------------------------------------------
            # ACCESSORY EXCLUSION
            # -----------------------------------------------------

            if (
                main_device_requested
                and not user_asks_accessory
            ):

                if leaf in accessory_leaf_categories:
                    continue

                if any(
                    acc in main_p
                    for acc in accessory_nouns
                ):
                    continue

            # -----------------------------------------------------
            # PRODUCT TYPE VERIFICATION
            # -----------------------------------------------------

            if requested_types:

                type_matched = False

                for req_type in requested_types:

                    valid_keywords = product_type_map[
                        req_type
                    ]

                    cat_match = any(
                        keyword in leaf
                        or keyword in cat_leaves
                        for keyword in valid_keywords
                    )

                    title_match = any(
                        keyword in main_p
                        for keyword in valid_keywords
                    )

                    if req_type in {
                        "headphones",
                        "headphone",
                        "earphones"
                    }:

                        if (
                            leaf in {
                                "accessories",
                                "transmitters",
                                "receivers",
                                "adapters"
                            }
                            or "transmitter" in main_p
                            or "receiver" in main_p
                        ):

                            cat_match = False
                            title_match = False

                    if cat_match or title_match:

                        type_matched = True
                        break

                if not type_matched:
                    continue

            # -----------------------------------------------------
            # BRAND VERIFICATION
            # -----------------------------------------------------

            if brand_tokens:

                brand_matched = True

                for brand in brand_tokens:

                    if (
                        brand not in main_p
                        and brand not in c_lower
                    ):

                        brand_matched = False
                        break

                if not brand_matched:
                    continue

            # -----------------------------------------------------
            # SCORE
            # -----------------------------------------------------

            score = 1000

            for modifier in modifiers:

                if (
                    modifier in main_p
                    or modifier in c_lower
                ):

                    score += 500

            if query.lower() in main_p:

                score += 2000

            # -----------------------------------------------------
            # REVIEW TEXT
            # -----------------------------------------------------

            text_parts = []

            if (
                review_title
                and review_title.lower() != "nan"
            ):

                text_parts.append(review_title)

            if (
                review_content
                and review_content.lower() != "nan"
            ):

                text_parts.append(review_content)

            text = " ".join(text_parts)

            # -----------------------------------------------------
            # SENTIMENT FROM RATING
            # -----------------------------------------------------

            sentiment = ""

            try:

                numeric_rating = float(rating)

                if numeric_rating >= 4:
                    sentiment = "positive"

                elif numeric_rating == 3:
                    sentiment = "neutral"

                elif numeric_rating < 3:
                    sentiment = "negative"

            except (
                ValueError,
                TypeError
            ):

                sentiment = ""

            # -----------------------------------------------------
            # STANDARDIZED RECORD
            # -----------------------------------------------------

            record = {
                "source": "amazon",

                "title": product_name,

                "text": text,

                "url": product_link,

                "timestamp": "",

                "rating": rating,

                "category": category,

                "sentiment": sentiment,

                "product_name": product_name,

                "asin": product_id
            }

            scored_records.append(
                (score, record)
            )

        scored_records.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return [
            item[1]
            for item in scored_records[:max_items]
        ]

    # =============================================================
    # NEW AMAZON ELECTRONICS DATASET SEARCH
    # =============================================================

    def _search_electronics_dataset(
        self,
        df: pd.DataFrame,
        query: str,
        tokens: List[str],
        max_items: int
    ) -> List[Dict[str, Any]]:

        if df.empty:
            return []

        results = []

        # ---------------------------------------------------------
        # SEARCH ELECTRONICS REVIEW DATASET
        #
        # Columns:
        # rating
        # title
        # text
        # images
        # asin
        # parent_asin
        # user_id
        # timestamp
        # helpful_vote
        # verified_purchase
        # review_sentiment
        # ---------------------------------------------------------

        for _, row in df.iterrows():

            review_title = str(
                row.get("title", "")
            ).strip()

            review_text = str(
                row.get("text", "")
            ).strip()

            asin = str(
                row.get("asin", "")
            ).strip()

            rating = str(
                row.get("rating", "")
            ).strip()

            sentiment = str(
                row.get("review_sentiment", "")
            ).strip()

            timestamp = str(
                row.get("timestamp", "")
            ).strip()

            verified_purchase = str(
                row.get("verified_purchase", "")
            ).strip()

            # -----------------------------------------------------
            # Combined searchable text
            # -----------------------------------------------------

            searchable_text = (
                f"{review_title} "
                f"{review_text}"
            ).lower()

            # -----------------------------------------------------
            # ALL QUERY TOKENS MUST MATCH
            #
            # This prevents:
            # "HP LAPTOP"
            # from matching records containing only "HP".
            # -----------------------------------------------------

            if tokens:

                if not all(
                    token in searchable_text
                    for token in tokens
                ):

                    continue

            # -----------------------------------------------------
            # SCORE
            # -----------------------------------------------------

            score = 500

            query_lower = query.lower()

            if query_lower in review_title.lower():
                score += 1500

            if query_lower in review_text.lower():
                score += 800

            for token in tokens:

                if token in review_title.lower():
                    score += 300

                if token in review_text.lower():
                    score += 100

            # Verified purchases get a small ranking boost
            if verified_purchase.lower() in {
                "true",
                "1",
                "yes"
            }:

                score += 50

            # -----------------------------------------------------
            # PRODUCT NAME
            #
            # This dataset does not contain a product_name column.
            # Use ASIN as the product identifier.
            # -----------------------------------------------------

            product_name = asin

            if not product_name:

                product_name = (
                    review_title
                    if review_title
                    else "Amazon Electronics Product"
                )

            # -----------------------------------------------------
            # STANDARDIZED RECORD
            # -----------------------------------------------------

            record = {
                "source": "amazon",

                "title": review_title
                if review_title
                else product_name,

                "text": review_text,

                "url": "",

                "timestamp": timestamp,

                "rating": rating,

                "category": "",

                "sentiment": sentiment,

                "product_name": product_name,

                "asin": asin,

                "verified_purchase": verified_purchase,

                "helpful_vote": str(
                    row.get("helpful_vote", "")
                ).strip(),

                "parent_asin": str(
                    row.get("parent_asin", "")
                ).strip()
            }

            results.append(
                (score, record)
            )

        results.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return [
            item[1]
            for item in results[:max_items]
        ]

    # =============================================================
    # MAIN FETCH FUNCTION
    # =============================================================

    def fetch_data(
        self,
        query: str,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:

        query = str(
            query or ""
        ).strip()

        if not query:
            return []

        print(
            f"[AmazonConnector] Searching Amazon datasets "
            f"for: '{query}'"
        )

        # ---------------------------------------------------------
        # QUERY TOKENS
        # ---------------------------------------------------------

        query_clean, tokens = (
            self._get_query_information(query)
        )

        # ---------------------------------------------------------
        # LOAD DATASETS
        # ---------------------------------------------------------

        old_df = self._load_old_dataset()

        electronics_df = (
            self._load_electronics_dataset()
        )

        # ---------------------------------------------------------
        # SEARCH BOTH DATASETS
        # ---------------------------------------------------------

        old_results = self._search_old_dataset(
            old_df,
            query_clean,
            tokens,
            max_items
        )

        print(
            "[AmazonConnector] Original dataset "
            f"matches: {len(old_results)}"
        )

        electronics_results = (
            self._search_electronics_dataset(
                electronics_df,
                query_clean,
                tokens,
                max_items
            )
        )

        print(
            "[AmazonConnector] Electronics dataset "
            f"matches: {len(electronics_results)}"
        )

        # ---------------------------------------------------------
        # COMBINE RESULTS
        # ---------------------------------------------------------

        combined = (
            old_results
            + electronics_results
        )

        # ---------------------------------------------------------
        # REMOVE DUPLICATES
        # ---------------------------------------------------------

        unique_results = []

        seen = set()

        for record in combined:

            key = (
                record.get("asin", ""),
                record.get("title", ""),
                record.get("text", "")
            )

            if key in seen:
                continue

            seen.add(key)

            unique_results.append(record)

            if len(unique_results) >= max_items:
                break

        # ---------------------------------------------------------
        # FINAL OUTPUT
        # ---------------------------------------------------------

        print(
            "[AmazonConnector] Combined search complete "
            f"for '{query}'. "
            f"Relevant records retrieved: "
            f"{len(unique_results)}"
        )

        return unique_results