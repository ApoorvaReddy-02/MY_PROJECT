from agents.preprocessing.agent import PreprocessingAgent


def test_clean_text():
    agent = PreprocessingAgent()

    text = "<p>Hello WORLD!</p> https://example.com"

    result = agent.clean_text(text)

    assert result == "hello world!"


def test_remove_empty_records():
    agent = PreprocessingAgent()

    records = [
        {"source": "amazon", "text": "Good product"},
        {"source": "reddit", "text": ""},
        {"source": "youtube", "text": "   "},
        {"source": "google", "text": "Excellent service"},
        {"source": "invalid"},
        "not a dictionary",
    ]

    result = agent.remove_empty_records(records)

    assert len(result) == 2
    assert result[0]["text"] == "Good product"
    assert result[1]["text"] == "Excellent service"


def test_process_adds_cleaned_text():
    agent = PreprocessingAgent()

    records = [
        {
            "source": "amazon",
            "text": "<p>This is a GREAT product!</p> https://example.com",
            "rating": "5",
        }
    ]

    result = agent.process(records)

    assert len(result) == 1
    assert result[0]["text"] == (
        "<p>This is a GREAT product!</p> https://example.com"
    )
    assert result[0]["cleaned_text"] == "this is a great product!"
    assert result[0]["rating"] == "5"


def test_duplicate_records_are_removed():
    agent = PreprocessingAgent()

    records = [
        {
            "source": "amazon",
            "text": "Great product!",
        },
        {
            "source": "amazon",
            "text": "Great product!",
        },
        {
            "source": "amazon",
            "text": "Bad product!",
        },
    ]

    result = agent.process(records)

    assert len(result) == 2
    assert result[0]["cleaned_text"] == "great product!"
    assert result[1]["cleaned_text"] == "bad product!"


def test_invalid_input_returns_empty_list():
    agent = PreprocessingAgent()

    result = agent.process(None)

    assert result == []


# ============================================================
# Connector-specific preprocessing tests
# ============================================================


def test_amazon_review():
    agent = PreprocessingAgent()

    records = [
        {
            "source": "amazon",
            "title": "Great Product",
            "text": "<p>This product is AMAZING!</p> https://amazon.com/item",
            "rating": "5",
            "asin": "B123",
        }
    ]

    result = agent.process(records)

    assert len(result) == 1
    assert result[0]["cleaned_text"] == "this product is amazing!"
    assert result[0]["rating"] == "5"
    assert result[0]["asin"] == "B123"


def test_reddit_comment():
    agent = PreprocessingAgent()

    records = [
        {
            "source": "reddit",
            "title": "Reddit Comment",
            "text": "I don't like this product :(",
            "category": "negative",
        }
    ]

    result = agent.process(records)

    assert len(result) == 1
    assert result[0]["cleaned_text"] == "i don't like this product :("
    assert result[0]["category"] == "negative"


def test_youtube_text():
    agent = PreprocessingAgent()

    records = [
        {
            "source": "youtube",
            "title": "iPhone Review",
            "text": "Amazing phone! https://youtube.com/watch?v=123",
        }
    ]

    result = agent.process(records)

    assert len(result) == 1
    assert result[0]["cleaned_text"] == "amazing phone!"
    assert result[0]["title"] == "iPhone Review"


def test_google_review():
    agent = PreprocessingAgent()

    records = [
        {
            "source": "google",
            "title": "ABC Restaurant",
            "text": "<div>The food was excellent!</div>",
            "rating": "5",
            "location": "Hyderabad",
        }
    ]

    result = agent.process(records)

    assert len(result) == 1
    assert result[0]["cleaned_text"] == "the food was excellent!"
    assert result[0]["rating"] == "5"
    assert result[0]["location"] == "Hyderabad"