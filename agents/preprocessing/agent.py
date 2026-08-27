"""
Preprocessing Agent
-------------------
Preprocessing Agent for the Agentic AI Sentiment Analysis System.

Receives standardized records from the Data Collection Agent
and prepares clean text for downstream agents.
"""

import re
from typing import Any, Dict, List


class PreprocessingAgent:
    """
    Cleans and prepares records received from the Data Collection Agent.
    """

    def clean_text(self, text: str) -> str:
        """
        Clean a single text value.

        Steps:
        1. Validate input
        2. Remove HTML tags
        3. Remove URLs
        4. Normalize whitespace
        5. Convert to lowercase
        """

        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Convert to lowercase
        text = text.lower()

        return text

    def remove_empty_records(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove records that do not contain usable text.
        """

        valid_records: List[Dict[str, Any]] = []

        for record in records:
            if not isinstance(record, dict):
                continue

            text = record.get("text", "")

            if isinstance(text, str) and text.strip():
                valid_records.append(record)

        return valid_records

    def remove_duplicate_records(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate records based on cleaned text.
        """

        unique_records: List[Dict[str, Any]] = []
        seen_text: set[str] = set()

        for record in records:
            cleaned_text = record.get("cleaned_text", "")

            if cleaned_text in seen_text:
                continue

            seen_text.add(cleaned_text)
            unique_records.append(record)

        return unique_records

    def process(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process records received from the Data Collection Agent.

        Original fields are preserved.
        A new 'cleaned_text' field is added.
        """

        if not isinstance(records, list):
            return []

        # Remove invalid/empty records
        valid_records = self.remove_empty_records(records)

        processed_records: List[Dict[str, Any]] = []

        # Clean each record
        for record in valid_records:
            processed_record = record.copy()

            original_text = record.get("text", "")
            cleaned_text = self.clean_text(original_text)

            # Skip if nothing remains after cleaning
            if not cleaned_text:
                continue

            processed_record["cleaned_text"] = cleaned_text
            processed_records.append(processed_record)

        # Remove duplicate cleaned text
        processed_records = self.remove_duplicate_records(
            processed_records
        )

        return processed_records