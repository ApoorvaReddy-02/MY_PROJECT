"""
preprocessing agent
-------------------
Preprocessing Agent for the Agentic AI Sentiment System.

Receives standardized records from the Data Collection Agent
and prepares clean text for downstream agents such as:

    Preprocessing Agent
            ↓
    Sentiment Analysis Agent
            ↓
    Trend Detection Agent
"""

import re
from typing import List, Dict, Any


class PreprocessingAgent:
    """
    Cleans and standardizes data received from the Data Collection Agent.
    """

    def clean_text(self, text: str) -> str:
        """
        Clean a single text value.

        Processing steps:
        1. Handle non-string values
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
            text
        )

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Convert to lowercase
        text = text.lower()

        return text

    def remove_empty_records(
        self,
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove records that do not contain usable text.
        """

        valid_records = []

        for record in records:

            text = record.get("text", "")

            if isinstance(text, str) and text.strip():
                valid_records.append(record)

        return valid_records

    def process(
        self,
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run the complete preprocessing pipeline.

        Input format from DataCollectionAgent:

        {
            "source": "...",
            "text": "...",
            "title": "...",
            "url": "...",
            "timestamp": "..."
        }

        Output keeps the original fields and adds:

        "cleaned_text": "..."
        """

        # Step 1: Remove records with empty text
        records = self.remove_empty_records(records)

        processed_records = []

        # Used to detect duplicate cleaned text
        seen_text = set()

        # Step 2: Process every record
        for record in records:

            # Keep the original record
            processed_record = record.copy()

            # Clean the text
            cleaned_text = self.clean_text(
                record.get("text", "")
            )

            # Skip if nothing remains after cleaning
            if not cleaned_text:
                continue

            # Remove duplicate cleaned text
            if cleaned_text in seen_text:
                continue

            seen_text.add(cleaned_text)

            # Add cleaned text without deleting original text
            processed_record["cleaned_text"] = cleaned_text

            processed_records.append(processed_record)

        return processed_records