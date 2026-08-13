"""
Streamlit Dashboard for:
Agentic AI System for Real-Time Sentiment Monitoring

Flow:
User Query -> DataCollectionAgent -> YouTubeConnector
-> Real YouTube Data -> DistilBERT -> Sentiment Dashboard
"""

import os
import sys
import torch
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.data_collection_agent import DataCollectionAgent


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "distilbert-sentiment-exp2"
)

ID2LABEL = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE"
}

LABEL_COLORS = {
    "NEGATIVE": "#E63946",
    "NEUTRAL": "#4A4E69",
    "POSITIVE": "#2A9D8F"
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic AI Sentiment Monitoring",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_sentiment_model(model_path: str):
    """Load and cache the fine-tuned DistilBERT model."""

    if not os.path.exists(model_path):
        st.error(
            f"Model path not found: {model_path}"
        )
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    return tokenizer, model


# ============================================================
# DATA COLLECTION AGENT
# ============================================================

@st.cache_resource
def load_data_collection_agent():
    """Create and cache the central DataCollectionAgent."""

    return DataCollectionAgent()


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text, tokenizer, model):
    """
    Run DistilBERT sentiment prediction.

    Returns:
        predicted_label
        confidence
        class_probabilities
    """

    if not text or not text.strip():
        return None, 0.0, {
            "NEGATIVE": 0.0,
            "NEUTRAL": 0.0,
            "POSITIVE": 0.0
        }

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_class_idx = torch.argmax(probabilities).item()

    confidence = probabilities[predicted_class_idx].item()

    class_probs = {
        ID2LABEL[idx]: float(probabilities[idx])
        for idx in range(len(ID2LABEL))
    }

    return (
        ID2LABEL[predicted_class_idx],
        confidence,
        class_probs
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "🤖 Agentic AI System for Real-Time Sentiment Monitoring"
    )

    st.caption(
        "Multi-source data collection + AI-powered sentiment analysis"
    )

    st.markdown("---")


    # --------------------------------------------------------
    # LOAD SYSTEM COMPONENTS
    # --------------------------------------------------------

    with st.spinner("Loading AI system..."):

        tokenizer, model = load_sentiment_model(
            MODEL_DIR
        )

        agent = load_data_collection_agent()


    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.header("System Architecture")

    st.sidebar.markdown(
        """
        **Central Agent**

        `DataCollectionAgent`

        **Connected Sources**

        - 📺 YouTube
        - 🟠 Reddit
        - 📰 News
        - 🛒 Amazon
        - 𝕏 X

        **AI Model**

        Fine-tuned DistilBERT

        **Classes**

        - NEGATIVE
        - NEUTRAL
        - POSITIVE
        """
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("Registered Sources")

    for source in agent.get_supported_sources():

        st.sidebar.write(
            f"✅ {source.upper()}"
        )

    st.sidebar.markdown("---")

    st.sidebar.info(
        "Live data collection is currently enabled "
        "for YouTube."
    )


    # --------------------------------------------------------
    # SEARCH SECTION
    # --------------------------------------------------------

    st.header("🔎 Real-Time Sentiment Search")

    st.write(
        "Enter a product, topic, or keyword. "
        "The Data Collection Agent will collect live "
        "YouTube data and analyze its sentiment."
    )


    search_query = st.text_input(
        "Search Product / Topic",
        placeholder="Example: iPhone 17",
        value="iPhone 17"
    )


    col1, col2 = st.columns([1, 5])

    with col1:

        collect_button = st.button(
            "🚀 Collect & Analyze",
            type="primary",
            use_container_width=True
        )


    # --------------------------------------------------------
    # COLLECTION + SENTIMENT ANALYSIS
    # --------------------------------------------------------

    if collect_button:

        if not search_query.strip():

            st.warning(
                "Please enter a product or topic."
            )

            return


        # ----------------------------------------------------
        # COLLECT DATA
        # ----------------------------------------------------

        with st.spinner(
            f"Collecting YouTube data for '{search_query}'..."
        ):

            try:

                records = agent.collect_data(
                    query=search_query,
                    source="youtube",
                    max_items=5
                )

            except Exception as error:

                st.error(
                    f"YouTube data collection failed: {error}"
                )

                return


        if not records:

            st.warning(
                "No YouTube data was found for this search."
            )

            return


        # ----------------------------------------------------
        # COLLECTION SUMMARY
        # ----------------------------------------------------

        st.markdown("---")

        st.header("📡 Data Collection Results")


        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Source",
            "YouTube"
        )

        c2.metric(
            "Items Collected",
            len(records)
        )

        c3.metric(
            "Search Query",
            search_query
        )


        # ----------------------------------------------------
        # ANALYZE EACH RECORD
        # ----------------------------------------------------

        analyzed_records = []


        with st.spinner(
            "Running sentiment analysis on collected data..."
        ):

            for record in records:

                text = record.get(
                    "text",
                    ""
                ).strip()

                title = record.get(
                    "title",
                    ""
                )

                # If description/text is empty,
                # use video title for sentiment analysis.

                analysis_text = text if text else title


                if not analysis_text:

                    continue


                sentiment, confidence, probabilities = (
                    predict_sentiment(
                        analysis_text,
                        tokenizer,
                        model
                    )
                )


                analyzed_records.append({

                    "source": record.get(
                        "source",
                        "youtube"
                    ),

                    "title": title,

                    "text": analysis_text,

                    "url": record.get(
                        "url",
                        ""
                    ),

                    "timestamp": record.get(
                        "timestamp",
                        ""
                    ),

                    "sentiment": sentiment,

                    "confidence": confidence,

                    "probabilities": probabilities
                })


        if not analyzed_records:

            st.warning(
                "Data was collected, but no usable text "
                "was available for sentiment analysis."
            )

            return


        # ----------------------------------------------------
        # SENTIMENT SUMMARY
        # ----------------------------------------------------

        st.markdown("---")

        st.header("📊 Sentiment Summary")


        negative_count = sum(
            1
            for r in analyzed_records
            if r["sentiment"] == "NEGATIVE"
        )

        neutral_count = sum(
            1
            for r in analyzed_records
            if r["sentiment"] == "NEUTRAL"
        )

        positive_count = sum(
            1
            for r in analyzed_records
            if r["sentiment"] == "POSITIVE"
        )


        total = len(analyzed_records)


        negative_percentage = (
            negative_count / total * 100
        )

        neutral_percentage = (
            neutral_count / total * 100
        )

        positive_percentage = (
            positive_count / total * 100
        )


        m1, m2, m3 = st.columns(3)


        m1.metric(
            "🔴 NEGATIVE",
            f"{negative_percentage:.1f}%"
        )

        m2.metric(
            "⚪ NEUTRAL",
            f"{neutral_percentage:.1f}%"
        )

        m3.metric(
            "🟢 POSITIVE",
            f"{positive_percentage:.1f}%"
        )


        # ----------------------------------------------------
        # PROGRESS BARS
        # ----------------------------------------------------

        st.write("### Overall Sentiment Distribution")


        st.write(
            f"**NEGATIVE — {negative_percentage:.1f}%**"
        )

        st.progress(
            negative_percentage / 100
        )


        st.write(
            f"**NEUTRAL — {neutral_percentage:.1f}%**"
        )

        st.progress(
            neutral_percentage / 100
        )


        st.write(
            f"**POSITIVE — {positive_percentage:.1f}%**"
        )

        st.progress(
            positive_percentage / 100
        )


        # ----------------------------------------------------
        # INDIVIDUAL RESULTS
        # ----------------------------------------------------

        st.markdown("---")

        st.header("🎥 Individual Collected Results")


        for index, record in enumerate(
            analyzed_records,
            start=1
        ):

            sentiment = record["sentiment"]

            confidence = record["confidence"]

            accent_color = LABEL_COLORS.get(
                sentiment,
                "#333333"
            )


            with st.container():

                st.markdown(
                    f"""
                    <div style="
                        border-left: 5px solid {accent_color};
                        padding: 12px 18px;
                        margin-bottom: 10px;
                        background-color: {accent_color}15;
                        border-radius: 6px;
                    ">
                    <h3 style="
                        margin: 0;
                        color: {accent_color};
                    ">
                    {index}. {record["title"]}
                    </h3>

                    <p>
                    <strong>Sentiment:</strong>
                    {sentiment}
                    &nbsp;&nbsp;|&nbsp;&nbsp;

                    <strong>Confidence:</strong>
                    {confidence * 100:.2f}%
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.write(
                    record["text"]
                )


                if record["url"]:

                    st.markdown(
                        f"[🔗 View on YouTube]({record['url']})"
                    )


                # Probability breakdown

                probs = record["probabilities"]

                p1, p2, p3 = st.columns(3)

                p1.write(
                    f"🔴 Negative: "
                    f"{probs['NEGATIVE'] * 100:.1f}%"
                )

                p2.write(
                    f"⚪ Neutral: "
                    f"{probs['NEUTRAL'] * 100:.1f}%"
                )

                p3.write(
                    f"🟢 Positive: "
                    f"{probs['POSITIVE'] * 100:.1f}%"
                )


                st.markdown("---")


    # --------------------------------------------------------
    # MANUAL ANALYSIS SECTION
    # --------------------------------------------------------

    with st.expander(
        "📝 Manual Text Analysis (Optional)"
    ):

        manual_text = st.text_area(
            "Enter text manually:",
            placeholder="Type a review here..."
        )


        if st.button(
            "Analyze Manual Text"
        ):

            if manual_text.strip():

                sentiment, confidence, probabilities = (
                    predict_sentiment(
                        manual_text,
                        tokenizer,
                        model
                    )
                )


                st.success(
                    f"Predicted Sentiment: "
                    f"{sentiment} "
                    f"({confidence * 100:.2f}% confidence)"
                )

            else:

                st.warning(
                    "Please enter some text."
                )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()