from agents.sentiment.agent import SentimentAnalysisAgent


def test_positive_sentiment():
    agent = SentimentAnalysisAgent()

    result = agent.analyze_text(
        "Samsung has an amazing camera and excellent battery life."
    )

    assert result["sentiment"] == "POSITIVE"
    assert 0.0 <= result["sentiment_score"] <= 1.0


def test_negative_sentiment():
    agent = SentimentAnalysisAgent()

    result = agent.analyze_text(
        "Samsung battery life is terrible and the phone is disappointing."
    )

    assert result["sentiment"] == "NEGATIVE"
    assert 0.0 <= result["sentiment_score"] <= 1.0


def test_empty_text():
    agent = SentimentAnalysisAgent()

    result = agent.analyze_text("")

    assert result["sentiment"] == "UNKNOWN"
    assert result["sentiment_score"] == 0.0


def test_process_records():
    agent = SentimentAnalysisAgent()

    records = [
        {
            "source": "youtube",
            "text": "Samsung has an amazing camera.",
            "cleaned_text": "samsung has an amazing camera.",
        },
        {
            "source": "youtube",
            "text": "The battery life is terrible.",
            "cleaned_text": "the battery life is terrible.",
        },
    ]

    results = agent.process(records)

    assert len(results) == 2

    assert results[0]["sentiment"] == "POSITIVE"
    assert results[1]["sentiment"] == "NEGATIVE"

    # Original fields must be preserved
    assert results[0]["source"] == "youtube"
    assert "cleaned_text" in results[0]

    # Sentiment fields must be added
    assert "sentiment" in results[0]
    assert "sentiment_score" in results[0]


def test_invalid_records():
    agent = SentimentAnalysisAgent()

    records = [
        None,
        {},
        {"source": "youtube"},
        {"cleaned_text": ""},
    ]

    results = agent.process(records)

    assert results == []