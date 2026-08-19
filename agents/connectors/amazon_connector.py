"""
amazon_connector.py
-------------------
Amazon Dataset Connector for the Data Collection Agent.

Uses ONLY the local Amazon CSV dataset (datasets/amazon.csv).
NO LIVE AMAZON API.

General-Purpose Semantic Product Search Engine:
- Extracts requested Product Type(s), Modifiers, and Brand/Subject tokens from any arbitrary query.
- Compares against product_name and category fields primarily.
- Enforces strict product-type verification: Excludes generic accessory items (cables, chargers, stands, cases, stylus, adapters, printers, mice)
  when the user query requests a primary device (phone, smartphone, iphone, laptop, notebook, tv, tablet).
- Strips compatibility clauses ('compatible for', 'compatible with', 'for iphone', 'for HP', etc.) so compatible accessories do not fake a brand match.
- If there is no genuine product-type match in the dataset, returns an empty list [] instead of misleading/unrelated fallback items.
"""

import os
import re
from typing import List, Dict, Any, Optional

import pandas as pd

from .base_connector import BaseConnector


class AmazonConnector(BaseConnector):
    """Amazon review dataset connector with general semantic product search."""

    def __init__(
        self,
        dataset_path: Optional[str] = None
    ):
        super().__init__(
            source_name="amazon",
            api_key=None
        )

        if dataset_path:
            self.dataset_path = dataset_path
        else:
            self.dataset_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "datasets",
                    "amazon.csv"
                )
            )

        print(f"[AmazonConnector] Dataset mode enabled. Path: {self.dataset_path}")

    def fetch_data(
        self,
        query: str,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform general semantic product search on local Amazon dataset.
        """

        query = str(query or "").strip()
        if not query:
            return []

        print(f"[AmazonConnector] Searching local dataset for: '{query}'")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"[AmazonConnector] Dataset not found:\n{self.dataset_path}"
            )

        try:
            df = pd.read_csv(
                self.dataset_path,
                encoding="utf-8",
                low_memory=False
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                self.dataset_path,
                encoding="latin1",
                low_memory=False
            )
        except Exception as e:
            raise RuntimeError(
                f"[AmazonConnector] Failed to read dataset: {e}"
            ) from e

        df.columns = [str(column).strip() for column in df.columns]
        df = df.fillna("")

        # -----------------------------------------------------------------
        # TAXONOMY & COMPATIBILITY PATTERNS
        # -----------------------------------------------------------------
        product_type_map = {
            "phone": {"smartphones", "basicmobiles", "mobiles", "cellphone", "smartphone"},
            "phones": {"smartphones", "basicmobiles", "mobiles", "cellphone", "smartphone"},
            "smartphone": {"smartphones", "basicmobiles", "mobiles", "cellphone"},
            "smartphones": {"smartphones", "basicmobiles", "mobiles", "cellphone"},
            "mobile": {"smartphones", "basicmobiles", "mobiles", "cellphone"},
            "iphone": {"smartphones", "basicmobiles", "mobiles", "cellphone", "iphone"},
            "laptop": {"laptops"},
            "laptops": {"laptops"},
            "notebook": {"laptops"},
            "headphones": {"headphones", "in-ear", "over-ear", "on-ear", "earbuds", "headsets"},
            "headphone": {"headphones", "in-ear", "over-ear", "on-ear", "earbuds", "headsets"},
            "earphones": {"headphones", "in-ear", "over-ear", "on-ear", "earbuds"},
            "speaker": {"bluetoothspeakers", "portablespeakers", "speakers", "soundbars"},
            "speakers": {"bluetoothspeakers", "portablespeakers", "speakers", "soundbars"},
            "cable": {"cables", "usbcables", "hdmicables"},
            "cables": {"cables", "usbcables", "hdmicables"},
            "mouse": {"mice", "mouse"},
            "mice": {"mice", "mouse"},
            "keyboard": {"keyboards"},
            "keyboards": {"keyboards"},
            "watch": {"smartwatches"},
            "smartwatch": {"smartwatches"},
            "tv": {"televisions", "smarttelevisions"},
            "television": {"televisions", "smarttelevisions"},
            "printer": {"printers"},
            "tablet": {"tablets"},
            "monitor": {"monitors"}
        }

        accessory_nouns = {
            "cable", "charger", "adapter", "caddy", "converter", "connector",
            "case", "cover", "glass", "protector", "cord", "wire", "stand",
            "bag", "sleeve", "holder", "table", "mount", "strap", "mat", "pad",
            "power bank", "stylus", "skin", "screen guard", "battery", "cartridge",
            "ink", "printer", "webcam", "mouse", "keyboard", "lapdesk", "fan",
            "cooling pad", "hub", "dock", "remote", "transmitter", "receiver"
        }

        accessory_leaf_categories = {
            "cables", "usbcables", "hdmicables", "chargers", "wallchargers",
            "powerbanks", "lapdesks", "laptopaccessories", "mice", "keyboards",
            "remotecontrols", "microsd", "memorycards", "cases", "covers",
            "screenprotectors", "stylus", "graphicstablets", "networkadapters",
            "wirelessusbadapters", "printers", "cleaning", "mousepad", "mousepads",
            "stationery", "officeproducts", "paper", "adapters", "transmitters",
            "receivers", "accessories", "mobileaccessories"
        }

        compat_delimiters = [
            "compatible for", "compatible with", "for iphone", "for samsung",
            "for laptop", "for pc", "for android", "for ipad", "case for",
            "cover for", "stand for", "cable for", "bag for", "sleeve for",
            "adapter for", "charger for", "protector for", "suitable for", "fits "
        ]

        # -----------------------------------------------------------------
        # QUERY PARSING & TOKEN EXTRACTION
        # -----------------------------------------------------------------
        query_clean = query.lower()
        raw_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", query_clean)
        stopwords = {
            "a", "an", "the", "and", "or", "for", "is", "of", "in", "on", "to",
            "with", "about", "at", "by", "from", "it", "this", "that"
        }
        tokens = [t for t in raw_tokens if t not in stopwords and len(t) > 1]
        if not tokens:
            tokens = raw_tokens

        requested_types = set()
        brand_tokens = []
        modifiers = []

        for t in tokens:
            if t in product_type_map:
                requested_types.add(t)
                if t == "iphone":
                    brand_tokens.append("iphone")
            elif t in {"gaming", "wireless", "bluetooth", "smart", "portable", "fast", "hd", "4k", "foldable"}:
                modifiers.append(t)
            else:
                brand_tokens.append(t)

        main_device_requested = any(
            rt in {"phone", "phones", "smartphone", "smartphones", "mobile", "iphone", "laptop", "laptops", "notebook", "tv", "television", "tablet"}
            for rt in requested_types
        )
        user_asks_accessory = any(t in accessory_nouns for t in tokens)

        scored_records = []

        # -----------------------------------------------------------------
        # SEARCH EXECUTION & SEMANTIC VERIFICATION
        # -----------------------------------------------------------------
        for _, row in df.iterrows():
            product_name = str(row.get("product_name", "")).strip()
            category = str(row.get("category", "")).strip()
            about_product = str(row.get("about_product", "")).strip()
            review_title = str(row.get("review_title", "")).strip()
            review_content = str(row.get("review_content", "")).strip()
            rating = str(row.get("rating", "")).strip()
            product_link = str(row.get("product_link", "")).strip()
            product_id = str(row.get("product_id", "")).strip()

            p_lower = product_name.lower()
            c_lower = category.lower()

            # Separate main title from compatibility text
            main_p = p_lower
            for delim in compat_delimiters:
                if delim in p_lower:
                    main_p = p_lower[:p_lower.find(delim)]
                    break

            cat_leaves = [part.strip().lower().replace("&", "").replace(" ", "") for part in c_lower.split("|")]
            leaf = cat_leaves[-1] if cat_leaves else c_lower

            # 1. MAIN DEVICE VS ACCESSORY EXCLUSION
            if main_device_requested and not user_asks_accessory:
                if leaf in accessory_leaf_categories:
                    continue
                if any(acc in main_p for acc in accessory_nouns):
                    continue

            # 2. PRODUCT TYPE VERIFICATION
            if requested_types:
                type_matched = False
                for req_type in requested_types:
                    valid_cat_keywords = product_type_map[req_type]
                    cat_match = any(ck in leaf or ck in cat_leaves for ck in valid_cat_keywords)
                    title_match = any(ck in main_p for ck in valid_cat_keywords)

                    if req_type in {"headphones", "headphone", "earphones"}:
                        if leaf in {"accessories", "transmitters", "receivers", "adapters"} or "transmitter" in main_p or "receiver" in main_p:
                            cat_match = False
                            title_match = False

                    if cat_match or title_match:
                        type_matched = True
                        break

                if not type_matched:
                    continue

            # 3. BRAND / SUBJECT VERIFICATION
            if brand_tokens:
                brand_matched = True
                for bt in brand_tokens:
                    if bt not in main_p and bt not in c_lower:
                        brand_matched = False
                        break
                if not brand_matched:
                    continue

            # 4. MODIFIER & SUBSTRING SCORING
            score = 1000
            for mod in modifiers:
                if mod in main_p or mod in c_lower:
                    score += 500

            if query_clean in main_p:
                score += 2000

            text_parts = []
            if review_title and review_title.lower() != "nan":
                text_parts.append(review_title)
            if review_content and review_content.lower() != "nan":
                text_parts.append(review_content)

            text = " ".join(text_parts)

            record = {
                "source": "amazon",
                "title": product_name,
                "text": text,
                "url": product_link,
                "timestamp": "",
                "rating": rating,
                "category": category,
                "product_name": product_name,
                "asin": product_id
            }

            scored_records.append((score, record))

        # Sort descending by score
        scored_records.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored_records[:max_items]]

        print(f"[AmazonConnector] Semantic search complete for '{query}'. Genuine matches retrieved: {len(results)}")
        return results