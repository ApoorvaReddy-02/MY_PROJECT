"""
bluesky_connector.py
--------------------
Bluesky Data Source Connector for the Data Collection Agent.

Uses local, real Bluesky post datasets (.jsonl) collected from Bluesky.
Searches across all Bluesky JSONL files in the datasets directory.

No generated/fake posts are used.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

from .base_connector import BaseConnector


class BlueskyConnector(BaseConnector):
    """Bluesky local-dataset connector."""

    def __init__(self):
        super().__init__(
            source_name="bluesky",
            api_key=None
        )

        self.datasets_dir = (
            Path(__file__).resolve().parents[2] / "datasets"
        )

    def _get_query_tokens(self, query: str) -> List[str]:
        """Extract meaningful search terms from the query."""

        query_clean = str(query or "").strip().lower()

        raw_tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query_clean
        )

        stopwords = {
            "a", "an", "the", "and", "or",
            "for", "is", "of", "in", "on",
            "to", "with", "about", "at",
            "by", "from", "this", "that"
        }

        tokens = [
            token
            for token in raw_tokens
            if token not in stopwords and len(token) > 1
        ]

        return tokens if tokens else raw_tokens

    def _get_dataset_files(self) -> List[Path]:
        """Return all Bluesky JSONL datasets."""

        if not self.datasets_dir.exists():
            return []

        return sorted(
            self.datasets_dir.glob("*.jsonl")
        )

    def _is_relevant(
        self,
        text: str,
        query: str,
        tokens: List[str]
    ) -> bool:
        """
        Strict relevance check.

        Exact phrase is accepted first.
        Otherwise every significant token must occur.
        """

        text_lower = text.lower()
        query_lower = query.lower()

        if query_lower in text_lower:
            return True

        if not tokens:
            return True

        return all(
            token in text_lower
            for token in tokens
        )

    def _is_fallback_relevant(
        self,
        text: str,
        token: str
    ) -> bool:
        """
        Relevance check for controlled fallback searches.

        The fallback token must actually occur in the post.
        """

        return token.lower() in text.lower()

    def _build_post_url(self, uri: str) -> str:
        """Convert an AT Protocol URI to a Bluesky web URL."""

        if not uri:
            return ""

        try:
            parts = uri.split("/")

            if len(parts) < 2:
                return ""

            did = parts[2]
            rkey = parts[-1]

            return (
                f"https://bsky.app/profile/"
                f"{did}/post/{rkey}"
            )

        except Exception:
            return ""

    def _read_posts(self, dataset_files: List[Path]):
        """
        Read all posts from all local Bluesky datasets.

        Returns parsed posts while avoiding duplicate URIs.
        """

        posts = []
        seen_uris = set()

        for dataset_file in dataset_files:

            print(
                f"[BlueskyConnector] Reading: "
                f"{dataset_file.name}"
            )

            try:
                with open(
                    dataset_file,
                    "r",
                    encoding="utf-8"
                ) as file:

                    for line in file:

                        line = line.strip()

                        if not line:
                            continue

                        try:
                            post = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        text = str(
                            post.get("text", "") or ""
                        ).strip()

                        if not text:
                            continue

                        uri = str(
                            post.get("uri", "") or ""
                        )

                        if uri and uri in seen_uris:
                            continue

                        if uri:
                            seen_uris.add(uri)

                        posts.append(post)

            except OSError as e:
                print(
                    f"[BlueskyConnector] Could not read "
                    f"{dataset_file.name}: {e}"
                )

        return posts

    def _standardize_post(
        self,
        post: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert a Bluesky post into the common format."""

        text = str(
            post.get("text", "") or ""
        ).strip()

        uri = str(
            post.get("uri", "") or ""
        )

        created_at = str(
            post.get("created_at", "") or ""
        )

        author = str(
            post.get("author", "") or ""
        )

        return {
            "source": "bluesky",
            "text": text,
            "title": text[:100],
            "url": self._build_post_url(uri),
            "timestamp": created_at,
            "rating": "",
            "category": "",
            "product_name": "",
            "asin": "",
            "author": author
        }

    def _get_fallback_terms(
        self,
        query: str
    ) -> List[str]:
        """
        Return controlled fallback terms.

        These are only used when the strict search
        does not provide enough results.
        """

        query_lower = query.lower().strip()

        fallback_map = {
            "hp laptop": [
                "hp laptop",
                "hp",
                "laptop"
            ],
            "earphones": [
                "earphones",
                "earphone",
                "headphones",
                "headphone"
            ],
            "earphone": [
                "earphone",
                "earphones",
                "headphones",
                "headphone"
            ]
        }

        return fallback_map.get(
            query_lower,
            []
        )

    def fetch_data(
        self,
        query: str,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:

        query = str(query or "").strip()

        if not query:
            return []

        if max_items <= 0:
            return []

        dataset_files = self._get_dataset_files()

        if not dataset_files:
            raise FileNotFoundError(
                "[BlueskyConnector] No Bluesky .jsonl "
                "datasets found in: "
                f"{self.datasets_dir}"
            )

        tokens = self._get_query_tokens(query)

        print(
            "\n[BlueskyConnector] "
            "Searching local Bluesky datasets..."
        )

        print(f"Query: {query}")
        print(f"Dataset files: {len(dataset_files)}")

        # ---------------------------------------------------------
        # READ DATASETS ONCE
        # ---------------------------------------------------------

        posts = self._read_posts(dataset_files)

        # ---------------------------------------------------------
        # STEP 1: STRICT SEARCH
        # ---------------------------------------------------------

        results = []
        result_uris = set()

        for post in posts:

            if len(results) >= max_items:
                break

            text = str(
                post.get("text", "") or ""
            ).strip()

            if not text:
                continue

            if not self._is_relevant(
                text,
                query,
                tokens
            ):
                continue

            uri = str(
                post.get("uri", "") or ""
            )

            if uri and uri in result_uris:
                continue

            if uri:
                result_uris.add(uri)

            results.append(
                self._standardize_post(post)
            )

        print(
            "[BlueskyConnector] "
            f"Strict matches: {len(results)}"
        )

        # ---------------------------------------------------------
        # STEP 2: CONTROLLED FALLBACK
        # ---------------------------------------------------------

        if len(results) < max_items:

            fallback_terms = self._get_fallback_terms(
                query
            )

            if fallback_terms:

                print(
                    "[BlueskyConnector] "
                    "Strict search returned fewer "
                    "than requested."
                )

                print(
                    "[BlueskyConnector] "
                    f"Fallback terms: {fallback_terms}"
                )

                for fallback_term in fallback_terms:

                    if len(results) >= max_items:
                        break

                    print(
                        "[BlueskyConnector] "
                        f"Fallback search: "
                        f"{fallback_term}"
                    )

                    for post in posts:

                        if len(results) >= max_items:
                            break

                        text = str(
                            post.get("text", "") or ""
                        ).strip()

                        if not text:
                            continue

                        if not self._is_fallback_relevant(
                            text,
                            fallback_term
                        ):
                            continue

                        uri = str(
                            post.get("uri", "") or ""
                        )

                        if uri and uri in result_uris:
                            continue

                        if uri:
                            result_uris.add(uri)

                        results.append(
                            self._standardize_post(post)
                        )

        # ---------------------------------------------------------
        # FINAL RESULT
        # ---------------------------------------------------------

        results = results[:max_items]

        print(
            "[BlueskyConnector] Collection complete. "
            f"Relevant records retrieved: {len(results)}"
        )

        return results