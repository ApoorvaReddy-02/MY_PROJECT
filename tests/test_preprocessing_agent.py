from agents.preprocessing.agent import PreprocessingAgent


def test_clean_text():
    agent = PreprocessingAgent()

    text = "<p>I LOVE this product!</p> https://example.com"

    result = agent.clean_text(text)

    assert result == "i love this product!"


def test_remove_empty_records():
    agent = PreprocessingAgent()

    records = [
        {"text": "Good product"},
        {"text": ""},
        {"text": "   "},
        {"text": None},
        "invalid record",
    ]

    result = agent.remove_empty_records(records)

    assert len(result) == 1
    assert result[0]["text"] == "Good product"


def test_process_preserves_original_fields():
    agent = PreprocessingAgent()

    records = [
        {
            "source": "youtube",
            "text": "  I LOVE this product!  ",
            "title": "Product Review",
            "url": "https://youtube.com/example",
            "timestamp": "2026-08-26",
            "rating": "",
            "category": "",
            "product_name": "Example Product",
            "asin": "",
        }
    ]

    result = agent.process(records)

    assert len(result) == 1

    assert result[0]["text"] == "  I LOVE this product!  "
    assert result[0]["source"] == "youtube"
    assert result[0]["title"] == "Product Review"

    assert result[0]["cleaned_text"] == "i love this product!"


def test_remove_duplicates():
    agent = PreprocessingAgent()

    records = [
        {"source": "youtube", "text": "I love this product"},
        {"source": "reddit", "text": "I love this product"},
        {"source": "amazon", "text": "Amazing product"},
    ]

    result = agent.process(records)

    assert len(result) == 2
    assert result[0]["cleaned_text"] == "i love this product"
    assert result[1]["cleaned_text"] == "amazing product"


def test_invalid_input():
    agent = PreprocessingAgent()

    assert agent.process(None) == []
    assert agent.process({}) == []
    assert agent.process("invalid") == []