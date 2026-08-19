import os
import re
from urllib.parse import urljoin

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from agents.data_collection_agent import DataCollectionAgent


# =========================================================
# MODEL CONFIG
# =========================================================
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "distilbert-sentiment-exp2"
)

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


@st.cache_resource
def load_sentiment_model():
    """Load and cache the fine-tuned 3-class DistilBERT model."""
    if not os.path.exists(MODEL_DIR):
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading fine-tuned DistilBERT model: {e}")
        return None, None


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Agentic AI Sentiment Monitoring",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.15rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        background: white;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }

    .review-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        margin-bottom: 12px;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SENTIMENT HELPERS
# =========================================================
def normalize_sentiment(value):
    value = str(value or "").strip().lower()

    mapping = {
        "positive": "positive",
        "pos": "positive",
        "1": "positive",
        "negative": "negative",
        "neg": "negative",
        "-1": "negative",
        "neutral": "neutral",
        "neu": "neutral",
        "0": "neutral",
        "mixed": "neutral",
    }

    return mapping.get(value, "")


def sentiment_from_rating(rating):
    try:
        value = float(str(rating).strip())
    except (ValueError, TypeError):
        return ""

    if value >= 4:
        return "positive"
    if value == 3:
        return "neutral"
    if value <= 2:
        return "negative"

    return ""


def simple_text_sentiment(text):
    """Rule-based text fallback when model is unavailable."""
    text = str(text or "").lower()

    if not text.strip():
        return "neutral"

    positive_words = {
        "good", "great", "excellent", "amazing", "awesome", "love",
        "loved", "best", "worth", "recommend", "recommended", "happy",
        "perfect", "nice", "useful", "durable", "fast", "helpful",
        "satisfied", "fantastic", "wonderful", "works", "working"
    }

    negative_words = {
        "bad", "poor", "worst", "hate", "hated", "terrible", "awful",
        "waste", "broken", "slow", "useless", "disappointed",
        "disappointing", "problem", "problems", "issue", "issues",
        "defective", "damage", "damaged", "not working", "fails",
        "failure", "unhappy", "expensive"
    }

    positive_score = sum(
        1 for word in positive_words
        if re.search(r"\b" + re.escape(word) + r"\b", text)
    )

    negative_score = sum(
        1 for word in negative_words
        if re.search(r"\b" + re.escape(word) + r"\b", text)
    )

    if positive_score > negative_score:
        return "positive"

    if negative_score > positive_score:
        return "negative"

    return "neutral"


def ai_text_sentiment(text):
    """Runs inference using the fine-tuned 3-class DistilBERT model."""
    text = str(text or "").strip()
    if not text:
        return "neutral"

    tokenizer, model = load_sentiment_model()
    if tokenizer is None or model is None:
        return simple_text_sentiment(text)

    try:
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()

        return ID2LABEL.get(predicted_class_idx, "neutral")
    except Exception as e:
        print(f"DistilBERT inference error: {e}")
        return simple_text_sentiment(text)


def get_record_sentiment(record):
    """
    Priority:
      1. Explicit sentiment
      2. Connector category
      3. Star rating
      4. Fine-Tuned DistilBERT AI Model
    """
    sentiment = normalize_sentiment(record.get("sentiment", ""))
    if sentiment:
        return sentiment

    sentiment = normalize_sentiment(record.get("category", ""))
    if sentiment:
        return sentiment

    sentiment = sentiment_from_rating(record.get("rating", ""))
    if sentiment:
        return sentiment

    text = " ".join(
        [
            str(record.get("title", "") or ""),
            str(record.get("text", "") or ""),
        ]
    ).strip()

    res = ai_text_sentiment(text)
    return res if res else "neutral"


# =========================================================
# REVIEW URL HELPERS
# =========================================================
def extract_amazon_asin(row):
    candidates = [
        row.get("asin", ""),
        row.get("product_id", ""),
        row.get("product_name", ""),
        row.get("url", ""),
        row.get("review_url", ""),
        row.get("product_link", ""),
    ]

    for value in candidates:
        value = str(value or "").strip().upper()
        if not value:
            continue

        # Direct 10-character ASIN (e.g., B07JW9H4J1)
        if len(value) == 10 and value.startswith("B") and value.isalnum():
            return value

        # /dp/B0XXXXXXXXX
        match = re.search(r"/DP/([A-Z0-9]{10})", value)
        if match:
            return match.group(1)

        # /PRODUCT-REVIEWS/B0XXXXXXXXX
        match = re.search(r"/PRODUCT-REVIEWS/([A-Z0-9]{10})", value)
        if match:
            return match.group(1)

        # Standalone ASIN match in string
        match = re.search(r"\bB[A-Z0-9]{9}\b", value)
        if match:
            return match.group(0)

    return ""


def get_review_url(row):
    source = str(row.get("source", "") or "").strip().lower()

    # Amazon: construct direct product-reviews URL from ASIN or product_link
    if source == "amazon":
        asin = extract_amazon_asin(row)
        if asin:
            return f"https://www.amazon.in/product-reviews/{asin}/"

    # Best case: connector gives an exact review URL.
    review_url = str(row.get("review_url", "") or "").strip()
    if review_url:
        return review_url

    url = str(row.get("url", "") or "").strip()

    # Flipkart: use the connector URL if it exists.
    if source == "flipkart":
        if url.startswith("/"):
            return urljoin("https://www.flipkart.com", url)
        return url

    # YouTube / Reddit / News / X
    return url


# =========================================================
# CLEAR SEARCH
# =========================================================
def clear_search():
    st.session_state.global_query = ""
    st.session_state.records = []
    st.session_state.last_source = ""
    st.session_state.last_query = ""


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">🤖 Agentic AI System for Real-Time Sentiment Monitoring</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Multi-source data collection • Sentiment analysis • Trend monitoring • Decision support</div>',
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================
if "records" not in st.session_state:
    st.session_state.records = []

if "last_source" not in st.session_state:
    st.session_state.last_source = ""

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "global_query" not in st.session_state:
    st.session_state.global_query = ""


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Data Collection")

source = st.sidebar.selectbox(
    "Select Source",
    ["amazon", "flipkart", "youtube", "reddit", "news", "x"],
)

# IMPORTANT:
# ONE query box is used for every source.
# Therefore HP LAPTOP remains when switching Amazon -> Flipkart -> YouTube.
query = st.sidebar.text_input(
    "Search / Query",
    key="global_query",
    placeholder="Example: HP LAPTOP, iPhone, headphones...",
)

clear_col, info_col = st.sidebar.columns([1, 1])

with clear_col:
    st.button(
        "🧹 Clear",
        use_container_width=True,
        on_click=clear_search,
    )

with info_col:
    st.caption("Same search")

max_items = st.sidebar.slider(
    "Maximum Records",
    min_value=1,
    max_value=50,
    value=10,
)

collect = st.sidebar.button(
    "🚀 Collect & Analyze",
    type="primary",
    use_container_width=True,
)

# Helpful status for Flipkart.
if source == "flipkart" and not os.getenv("PARSE_API_KEY"):
    st.sidebar.warning(
        "Flipkart needs PARSE_API_KEY for the current connector."
    )


# =========================================================
# COLLECTION
# =========================================================
if collect:
    search_query = st.session_state.global_query.strip()

    if not search_query:
        st.error("Please enter a search query first.")
    else:
        with st.spinner(f"Collecting data from {source.title()}..."):
            try:
                agent = DataCollectionAgent()

                records = agent.collect_data(
                    query=search_query,
                    source=source,
                    max_items=max_items,
                )

                st.session_state.records = records or []
                st.session_state.last_source = source
                st.session_state.last_query = search_query

                if not records:
                    st.warning(
                        f"No matching results found for '{search_query}' on {source.title()}."
                    )

            except Exception as exc:
                st.session_state.records = []

                if source == "flipkart" and "PARSE_API_KEY" in str(exc):
                    st.error(
                        "Flipkart collection needs your PARSE_API_KEY. "
                        "The dashboard is working, but the Flipkart connector "
                        "cannot collect data without its API credentials."
                    )
                else:
                    st.error(f"Data collection failed: {exc}")


# =========================================================
# MAIN DASHBOARD
# =========================================================
records = st.session_state.records

if not records:
    if st.session_state.last_query and st.session_state.last_source:
        st.info(
            f"No matching results found for **'{st.session_state.last_query}'** on **{st.session_state.last_source.title()}**."
        )
    else:
        st.info(
            "👈 Enter one product/keyword (e.g. **HP LAPTOP**), choose a source, and click "
            "**Collect & Analyze**."
        )

    st.markdown("### System Flow")
    st.code(
        """Data Collection Agent
        ↓
Source Filtering / Cleaning
        ↓
NLP Preprocessing
        ↓
Sentiment Analysis
        ↓
Trend Detection
        ↓
Dashboard & Decision Support""",
        language="text",
    )

else:
    df = pd.DataFrame(records)

    # -----------------------------------------------------
    # ENSURE STANDARD COLUMNS
    # -----------------------------------------------------
    expected_columns = [
        "source",
        "author",
        "title",
        "text",
        "url",
        "review_url",
        "timestamp",
        "rating",
        "category",
        "sentiment",
        "product_name",
        "asin",
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""

    # -----------------------------------------------------
    # CLEAN VALUES
    # -----------------------------------------------------
    for col in ["title", "text", "rating", "timestamp", "product_name"]:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["text"] = df["text"].replace(
        {
            "None": "",
            "None None": "",
            "nan": "",
        }
    )

    # -----------------------------------------------------
    # SENTIMENT
    # -----------------------------------------------------
    # This fixes the "unknown" problem for records that have
    # category, rating, or review text.
    df["sentiment"] = [
        get_record_sentiment(record)
        for record in records
    ]

    total = len(df)
    positive = int((df["sentiment"] == "positive").sum())
    negative = int((df["sentiment"] == "negative").sum())
    neutral = int((df["sentiment"] == "neutral").sum())

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{positive}</div>
                <div class="metric-label">Positive</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{negative}</div>
                <div class="metric-label">Negative</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{neutral}</div>
                <div class="metric-label">Neutral</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        f"Source: **{st.session_state.last_source}**  |  "
        f"Query: **{st.session_state.last_query}**"
    )

    # -----------------------------------------------------
    # CHARTS
    # -----------------------------------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("📊 Sentiment Distribution")

        sentiment_counts = (
            df["sentiment"]
            .value_counts()
            .reindex(
                ["positive", "negative", "neutral"],
                fill_value=0,
            )
        )

        st.bar_chart(sentiment_counts)

    with right:
        st.subheader("⭐ Rating Distribution")

        ratings = pd.to_numeric(
            df["rating"],
            errors="coerce",
        ).dropna()

        if len(ratings):
            rating_counts = (
                ratings
                .value_counts()
                .sort_index()
            )
            st.bar_chart(rating_counts)
        else:
            st.info("No numeric rating data available.")

    # -----------------------------------------------------
    # PLATFORM SUMMARY
    # -----------------------------------------------------
    st.subheader("🌐 Platform Summary")

    source_summary = (
        df.groupby(["source", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    st.dataframe(
        source_summary,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # REVIEWS / POSTS
    # -----------------------------------------------------
    st.subheader("📝 Collected Reviews / Posts")

    for idx, row in df.iterrows():

        title = str(row.get("title", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        source_name = str(row.get("source", "") or "").strip()
        sentiment = str(row.get("sentiment", "") or "").strip().lower()
        rating = str(row.get("rating", "") or "").strip()
        timestamp = str(row.get("timestamp", "") or "").strip()

        if not title:
            title = "User Review / Post"

        if not text:
            text = "Review text was not provided by the source."

        st.markdown(
            f"""
            <div class="review-card">
                <b>{title}</b><br>
                <small>
                    Source: {source_name}
                    &nbsp; | &nbsp;
                    Sentiment: <b>{sentiment.title()}</b>
                    &nbsp; | &nbsp;
                    Rating: {rating or "N/A"}
                </small>
                <p>{text}</p>
                <small>{timestamp}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # ORIGINAL REVIEW LINK
        # -------------------------------------------------
        review_url = get_review_url(row)

        if review_url:
            if source_name.lower() == "amazon":
                button_text = "🔗 View Original Amazon Reviews"
            elif source_name.lower() == "flipkart":
                button_text = "🔗 View Original Flipkart Page"
            else:
                button_text = f"🔗 View Original {source_name.title()}"

            st.link_button(
                button_text,
                review_url,
                key=f"review_link_{idx}",
            )
        else:
            st.caption(
                f"⚠️ No original {source_name.title()} URL was provided."
            )

    # -----------------------------------------------------
    # RAW DATA
    # -----------------------------------------------------
    with st.expander("🔍 View standardized records"):
        raw_columns = [
            "source",
            "author",
            "title",
            "text",
            "url",
            "review_url",
            "timestamp",
            "rating",
            "sentiment",
            "category",
            "product_name",
            "asin",
        ]

        st.dataframe(
            df[raw_columns],
            use_container_width=True,
        )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------
    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Results as CSV",
        data=csv_data,
        file_name="sentiment_results.csv",
        mime="text/csv",
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Agentic AI System for Real-Time Sentiment Monitoring and Decision Support"
)