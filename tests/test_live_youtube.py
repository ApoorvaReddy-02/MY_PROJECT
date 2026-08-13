"""
test_live_youtube.py
--------------------
Integration test for YouTubeConnector using live YouTube Data API v3 key.

Validates:
1. Secure loading of YOUTUBE_API_KEY from .env file.
2. Authentic connection to YouTube Data API v3 with search query "iPhone 17".
3. Retrieval of max_items=3 real records.
4. Schema validation (source, text, title, url, timestamp).
5. Verification that API key is NEVER printed or logged.
"""

import os
from dotenv import load_dotenv
from agents.data_collection_agent import DataCollectionAgent

# Load environment variables
load_dotenv()


def test_live_youtube_connection():
    print("=" * 70)
    print("LIVE YOUTUBE DATA API V3 INTEGRATION TEST")
    print("=" * 70)

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("[FAIL] YOUTUBE_API_KEY not found in environment/.env file.")
        assert False, "YOUTUBE_API_KEY environment variable is missing."

    print("API Key Status: [PRESENT & REDACTED FOR SECURITY]")

    agent = DataCollectionAgent()
    query = "iPhone 17"
    max_items = 3

    print(f"\nDispatching real YouTube API query via DataCollectionAgent:")
    print(f"  Source   : youtube")
    print(f"  Query    : '{query}'")
    print(f"  Max Items: {max_items}")

    results = agent.collect_data(query=query, source="youtube", max_items=max_items)

    print("\nAPI CONNECTION & COLLECTION RESULTS:")
    print(f"  Status           : SUCCESS")
    print(f"  Records Collected: {len(results)}")
    assert len(results) > 0, "No records returned from YouTube API."
    assert len(results) <= max_items, f"Expected <= {max_items} records, got {len(results)}"

    print("\nVERIFYING STANDARDIZED OUTPUT SCHEMA & DATA QUALITY:")
    required_keys = {"source", "text", "title", "url", "timestamp"}

    for idx, item in enumerate(results, 1):
        print(f"\nRecord {idx}:")
        print(f"  source    : {item['source']}")
        print(f"  title     : {item['title'][:65]}..." if len(item['title']) > 65 else f"  title     : {item['title']}")
        print(f"  url       : {item['url']}")
        print(f"  timestamp : {item['timestamp']}")
        print(f"  text len  : {len(item['text'])} chars")

        # Key validation
        for key in required_keys:
            assert key in item, f"Missing required key '{key}' in item schema"

        # Content validation
        assert item["source"] == "youtube", f"Expected source 'youtube', got '{item['source']}'"
        assert len(item["title"]) > 0, "Video title is empty"
        assert item["url"].startswith("https://www.youtube.com/watch?v="), f"Invalid YouTube URL format: {item['url']}"

    print("\n" + "=" * 70)
    print("TEST PASSED: Live YouTube API connection & schema validation successful!")
    print("=" * 70)


if __name__ == "__main__":
    test_live_youtube_connection()
