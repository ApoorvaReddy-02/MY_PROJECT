from agents.preprocessing.agent import PreprocessingAgent


def test_preprocessing_agent():

    # Sample data matching the output
    # of the Data Collection Agent
    records = [
        {
            "source": "youtube",
            "text": "  I LOVE this product!!! https://example.com  ",
            "title": "Product Review",
            "url": "https://youtube.com/example",
            "timestamp": "2026-08-13T10:00:00"
        },
        {
            "source": "youtube",
            "text": "",
            "title": "Empty Review",
            "url": "https://youtube.com/example2",
            "timestamp": "2026-08-13T10:01:00"
        },
        {
            "source": "youtube",
            "text": "I LOVE this product!!!",
            "title": "Duplicate Review",
            "url": "https://youtube.com/example3",
            "timestamp": "2026-08-13T10:02:00"
        }
    ]

    # Create preprocessing agent
    agent = PreprocessingAgent()

    # Process the records
    result = agent.process(records)

    # Empty record should be removed
    assert len(result) == 1

    # Original fields should still exist
    assert result[0]["source"] == "youtube"
    assert result[0]["title"] == "Product Review"
    assert result[0]["url"] == "https://youtube.com/example"
    assert result[0]["timestamp"] == "2026-08-13T10:00:00"

    # Original text should remain unchanged
    assert result[0]["text"] == (
        "  I LOVE this product!!! https://example.com  "
    )

    # Cleaned text should be created
    assert result[0]["cleaned_text"] == "i love this product!!!"