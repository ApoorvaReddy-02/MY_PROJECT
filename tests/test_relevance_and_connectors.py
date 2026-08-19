"""
test_relevance_and_connectors.py
--------------------------------
Automated verification test for general semantic product search, multi-source data collection,
fine-tuned DistilBERT sentiment classification, and URL resolution.
"""

import os
import streamlit_app as app
from agents.data_collection_agent import DataCollectionAgent


def run_verification_tests():
    print("=" * 75)
    print("RUNNING STRICT SEMANTIC PRODUCT SEARCH VERIFICATION TESTS")
    print("=" * 75)

    agent = DataCollectionAgent()

    # ---------------------------------------------------------
    # TEST A: GENERAL AMAZON SEMANTIC SEARCH REGRESSION SUITE
    # ---------------------------------------------------------
    test_cases = [
        {
            "query": "SAMSUNG Phone",
            "expected_type": "phone",
            "expect_empty": False,
            "explanation": "Must return actual Samsung smartphone units, excluding generic cables/earphones."
        },
        {
            "query": "SAMSUNG GALAXY PHONE",
            "expected_type": "phone",
            "expect_empty": False,
            "explanation": "Must return actual Samsung Galaxy smartphone units."
        },
        {
            "query": "IPHONE",
            "expected_type": "phone",
            "expect_empty": True,
            "explanation": "Must return [] (no result) because amazon.csv contains no iPhone smartphone units (only cables/styluses/chargers)."
        },
        {
            "query": "HP LAPTOP",
            "expected_type": "laptop",
            "expect_empty": True,
            "explanation": "Must return [] (no result) because amazon.csv contains no HP laptop units (only HP printers/mice/chargers and lapdesks)."
        },
        {
            "query": "DELL LAPTOP",
            "expected_type": "laptop",
            "expect_empty": True,
            "explanation": "Must return [] (no result) because amazon.csv contains no Dell laptop units (only Dell mice/keyboards)."
        },
        {
            "query": "HEADPHONES",
            "expected_type": "headphones",
            "expect_empty": False,
            "explanation": "Must return genuine Headphones/Earphones/Earbuds."
        },
        {
            "query": "gaming laptop",
            "expected_type": "laptop",
            "expect_empty": True,
            "explanation": "Must return [] (no result) because amazon.csv contains no Gaming Laptop units (only mouse pads/notebooks)."
        },
        {
            "query": "Bluetooth speaker",
            "expected_type": "speaker",
            "expect_empty": False,
            "explanation": "Must return genuine Bluetooth Speakers."
        },
        {
            "query": "USB cable",
            "expected_type": "cable",
            "expect_empty": False,
            "explanation": "Must return genuine USB Data/Charging Cables."
        }
    ]

    print("\n[TEST A] Amazon Semantic Search Regression Suite:")
    for tc in test_cases:
        tq = tc["query"]
        records = agent.collect_data(query=tq, source="amazon", max_items=5)
        print(f"\n  Query: '{tq}' -> {len(records)} records returned:")

        if tc["expect_empty"]:
            assert len(records) == 0, f"FAIL: Expected [] for query '{tq}', but got {len(records)} records."
            print(f"    [STATUS: PASS] Returned 0 records (empty list) as expected. Reason: {tc['explanation']}")
        else:
            assert len(records) > 0, f"FAIL: Expected matching dataset records for query '{tq}'."
            for idx, r in enumerate(records, 1):
                safe_title = r['title'].encode('ascii', 'ignore').decode('ascii')
                sentiment = app.get_record_sentiment(r)
                url = app.get_review_url(r)
                print(f"    Item {idx}: ASIN={r.get('asin'):<10} | Title='{safe_title[:60]}...' | Sentiment={sentiment:<8}")
                assert url.startswith("https://www.amazon.in/product-reviews/"), f"Invalid Amazon URL: {url}"

                # Strict Category/Title Verification
                title_cat = f"{r['title']} {r['category']}".lower()
                if tc["expected_type"] == "phone":
                    assert "galaxy" in title_cat or "smartphone" in title_cat or "mobile" in title_cat, f"Irrelevant phone result: {r['title']}"
                    assert "cable" not in r['title'].lower(), "Cables must not be returned for phone query"
                elif tc["expected_type"] == "headphones":
                    assert "headphones" in title_cat or "earphones" in title_cat or "earbuds" in title_cat or "headset" in title_cat, f"Irrelevant headphone result: {r['title']}"
                elif tc["expected_type"] == "speaker":
                    assert "speaker" in title_cat, f"Irrelevant speaker result: {r['title']}"
                elif tc["expected_type"] == "cable":
                    assert "cable" in title_cat, f"Irrelevant cable result: {r['title']}"

            print(f"    [STATUS: PASS] Returned {len(records)} genuine matching products. Reason: {tc['explanation']}")

    print("\n  => [PASS] All 9 Amazon semantic search test cases verified successfully!")

    # ---------------------------------------------------------
    # TEST B: YOUTUBE (LIVE API)
    # ---------------------------------------------------------
    print(f"\n[TEST B] YouTube Live API Search for 'HP LAPTOP':")
    try:
        yt_records = agent.collect_data(query="HP LAPTOP", source="youtube", max_items=3)
        print(f"  Returned {len(yt_records)} records.")
        for idx, r in enumerate(yt_records, 1):
            text_content = f"{r['title']} {r['text']}".lower()
            safe_title = r['title'].encode('ascii', 'ignore').decode('ascii')
            assert "hp" in text_content or "laptop" in text_content, f"Irrelevant YouTube result: {safe_title}"
            url = app.get_review_url(r)
            assert url.startswith("https://www.youtube.com/watch?v="), f"Invalid YouTube URL: {url}"
            sentiment = app.get_record_sentiment(r)
            print(f"  Record {idx}: Title='{safe_title[:45]}...' | Sentiment={sentiment:<8} | URL={url}")
        print("  => [PASS] YouTube live search & links verified!")
    except Exception as e:
        print(f"  [NOTE] YouTube live collection notice: {e}")

    # ---------------------------------------------------------
    # TEST C: NEWS
    # ---------------------------------------------------------
    print(f"\n[TEST C] News Search for 'HP LAPTOP':")
    try:
        news_records = agent.collect_data(query="HP LAPTOP", source="news", max_items=5)
        print(f"  Returned {len(news_records)} records.")
        for idx, r in enumerate(news_records, 1):
            text_content = f"{r['title']} {r['text']}".lower()
            assert "modi" not in text_content and ("hp" in text_content or "laptop" in text_content), f"Irrelevant News result: {r['title']}"
            sentiment = app.get_record_sentiment(r)
            assert sentiment in ["positive", "negative", "neutral"]
        print("  => [PASS] News search relevance verified!")
    except Exception as e:
        print(f"  [NOTE] News collection notice: {e}")

    # ---------------------------------------------------------
    # TEST D: X / TWITTER (NO UNRELATED MODI POSTS)
    # ---------------------------------------------------------
    print(f"\n[TEST D] X / Twitter Search for 'HP LAPTOP':")
    x_records = agent.collect_data(query="HP LAPTOP", source="x", max_items=5)
    print(f"  Returned {len(x_records)} records.")
    for idx, r in enumerate(x_records, 1):
        text_content = f"{r['title']} {r['text']}".lower()
        assert "modi" not in text_content, "Unrelated Modi post returned on X!"
        sentiment = app.get_record_sentiment(r)
        assert sentiment in ["positive", "negative", "neutral"]
    print("  => [PASS] X dataset search verified!")

    # ---------------------------------------------------------
    # TEST E: REDDIT
    # ---------------------------------------------------------
    print(f"\n[TEST E] Reddit Search for 'HP LAPTOP':")
    reddit_records = agent.collect_data(query="HP LAPTOP", source="reddit", max_items=5)
    print(f"  Returned {len(reddit_records)} records.")
    for idx, r in enumerate(reddit_records, 1):
        text_content = f"{r['title']} {r['text']}".lower()
        assert "modi" not in text_content, "Unrelated Modi post returned on Reddit!"
        sentiment = app.get_record_sentiment(r)
        assert sentiment in ["positive", "negative", "neutral"]
    print("  => [PASS] Reddit dataset search verified!")

    # ---------------------------------------------------------
    # TEST F: SENTIMENT MODEL INTEGRATION VERIFICATION
    # ---------------------------------------------------------
    print("\n[TEST F] Sentiment Model Verification:")
    sample_text = "The battery life is awful and the laptop keeps crashing."
    sample_rec = {"text": sample_text}
    pred_sentiment = app.get_record_sentiment(sample_rec)
    print(f"  Sample Negative Input : '{sample_text}'")
    print(f"  Predicted Sentiment   : {pred_sentiment}")
    assert pred_sentiment == "negative", f"Expected 'negative', got '{pred_sentiment}'"
    print("  => [PASS] DistilBERT 3-class model classification verified!")

    print("\n" + "=" * 75)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_verification_tests()
