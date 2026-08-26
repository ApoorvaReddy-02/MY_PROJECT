from agents.data_collection_agent import DataCollectionAgent
from agents.preprocessing.agent import PreprocessingAgent


def test_data_collection_to_preprocessing():
    data_agent = DataCollectionAgent()
    preprocessing_agent = PreprocessingAgent()

    class MockConnector:
        def fetch_data(self, query, max_items):
            return [
                {
                    "text": "<p>I LOVE this product!</p> https://example.com",
                    "title": "Product Review",
                    "url": "https://example.com/review",
                    "timestamp": "2026-08-26",
                    "rating": "5",
                    "category": "electronics",
                    "product_name": "Test Product",
                    "asin": "TEST123",
                }
            ]

    data_agent.register_connector("test", MockConnector())

    collected_records = data_agent.collect_data(
        query="product",
        source="test",
        max_items=10,
    )

    assert len(collected_records) == 1

    processed_records = preprocessing_agent.process(
        collected_records
    )

    assert len(processed_records) == 1

    record = processed_records[0]

    # Data Collection fields are preserved
    assert record["source"] == "test"
    assert record["title"] == "Product Review"
    assert record["rating"] == "5"
    assert record["product_name"] == "Test Product"

    # Original text is preserved
    assert record["text"] == (
        "<p>I LOVE this product!</p> https://example.com"
    )

    # Preprocessing adds cleaned text
    assert record["cleaned_text"] == "i love this product!"