from agents.connectors.bluesky_connector import BlueskyConnector


connector = BlueskyConnector()

results = connector.fetch_data(
    query="iPhone 17",
    max_items=5
)

print(f"\nPosts found: {len(results)}\n")

for i, item in enumerate(results, start=1):
    print("=" * 70)
    print(f"POST {i}")
    print(f"Author: {item['author']}")
    print(f"Text: {item['text']}")
    print(f"URL: {item['url']}")
    print(f"Timestamp: {item['timestamp']}")