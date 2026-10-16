
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Customer Message Intelligence",
    page_icon="📩",
    layout="centered"
)

@st.cache_resource
def train_model():
    data = pd.read_csv("support_messages.csv")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(data["message"], data["category"])
    return model

st.title("📩 Customer Message Intelligence")
st.write(
    "Enter a customer message to find the most suitable "
    "support team."
)

try:
    model = train_model()

    message = st.text_area(
        "Customer message",
        placeholder="Example: My payment was deducted but my order failed"
    )

    if st.button("Classify Message", type="primary"):
        if message.strip():
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = model.classes_

            st.success(f"Recommended team: {prediction}")

            scores = pd.DataFrame({
                "Support team": classes,
                "Score": probabilities
            }).sort_values("Score", ascending=False)

            st.subheader("Prediction scores")
            st.bar_chart(scores.set_index("Support team"))

            st.caption(
                "Scores are model estimates, not guarantees "
                "that the prediction is correct."
            )
        else:
            st.warning("Please enter a customer message.")

except Exception as error:
    st.error(f"Unable to load the classifier: {error}")
