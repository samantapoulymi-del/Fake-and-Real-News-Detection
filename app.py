import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("fake_news_model.pkl")

model = load_model()

# Sidebar
st.sidebar.title("About")

st.sidebar.info(
    """
    Fake News Detection System

    Machine Learning Model:
    TF-IDF + Logistic Regression

    Features:
    - Fake/Real Prediction
    - Confidence Score
    - Prediction History
    """
)

# Title
st.title("📰 Fake News Detection System")

st.write(
    "Paste a news article below and click Analyze."
)

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Input Text
news_text = st.text_area(
    "Enter News Article",
    height=250
)

# Analyze Button
if st.button("Analyze News"):

    if len(news_text.strip()) == 0:

        st.warning("Please enter some news text.")

    else:

        prediction = model.predict([news_text])[0]

        probability = model.predict_proba(
            [news_text]
        )[0]

        confidence = max(probability)

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("🚨 Fake News Detected")
            else:
                st.success("✅ Real News")

        with col2:

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        st.subheader("Prediction Details")

        result_df = pd.DataFrame(
            {
                "Class": ["Real", "Fake"],
                "Probability": [
                    probability[0],
                    probability[1]
                ]
            }
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.progress(float(confidence))

        # Save history
        st.session_state.history.append(
            {
                "time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "prediction":
                "Fake" if prediction == 1 else "Real",
                "confidence":
                f"{confidence*100:.2f}%"
            }
        )

# History Section
st.divider()

st.subheader("Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

else:

    st.info(
        "No predictions made yet."
    )

# Footer
st.divider()

st.caption(
    "Educational Project | Fake News Detection using Machine Learning"
)