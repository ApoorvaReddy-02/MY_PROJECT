"""
test_data_collection_agent.py
------------------------------
Architecture Unit Test for DataCollectionAgent and BaseConnector.

Validates:
1. Dynamic registration of custom source connectors via BaseConnector interface.
2. Query dispatch from DataCollectionAgent to the registered connector.
3. Common output schema enforcement (source, text, title, url, timestamp).
"""

from typing import List, Dict, Any
from agents.connectors.base_connector import BaseConnector
from agents.data_collection_agent import DataCollectionAgent


class MockTestConnector(BaseConnector):
    """
    Mock/Test connector implementing BaseConnector for architectural validation.
    Used strictly for testing agent dispatch and schema enforcement.
    """

    def __init__(self):
        super().__init__(source_name="mock")

    def fetch_data(self, query: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """Returns structured sample test items to verify schema mapping."""
        mock_items = []
        for i in range(1, min(max_items, 3) + 1):
            mock_items.append({
                "text": f"Sample test content item #{i} for query '{query}'",
                "title": f"Test Document #{i}",
                "url": f"https://example.com/test/{i}",
                "timestamp": "2026-08-11T22:42:30Z"
            })
        return mock_items


def test_agent_architecture():
    print("=" * 60)
    print("DATA COLLECTION AGENT ARCHITECTURE TEST")
    print("=" * 60)

    # 1. Instantiate Central Agent
    agent = DataCollectionAgent()
    print("Default Registered Sources:", agent.get_supported_sources())

    # 2. Instantiate and Register Mock Connector
    mock_connector = MockTestConnector()
    agent.register_connector("mock", mock_connector)
    print("Updated Registered Sources:", agent.get_supported_sources())
    assert "mock" in agent.get_supported_sources(), "Mock connector failed to register."

    # 3. Dispatch Collection Request
    test_query = "agentic ai architecture test"
    results = agent.collect_data(query=test_query, source="mock", max_items=2)

    # 4. Verify Results and Schema
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    print("\nVERIFYING COMMON OUTPUT SCHEMA:")
    for idx, item in enumerate(results, 1):
        print(f"\nItem {idx}:")
        for key in ["source", "text", "title", "url", "timestamp"]:
            print(f"  {key:<10}: {item[key]}")
            assert key in item, f"Missing required key '{key}' in output schema."

        assert item["source"] == "mock", f"Expected source 'mock', got '{item['source']}'"
        assert test_query in item["text"], "Query string not passed correctly to connector."

    print("\n" + "=" * 60)
    print("TEST PASSED: DataCollectionAgent dispatch and schema enforcement verified!")
    print("=" * 60)


if __name__ == "__main__":
    test_agent_architecture()
