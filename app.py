import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Election Sentiment Analyzer")
st.title("🇮🇳 Election Tweet Sentiment Analyzer")

text = st.text_area("Type or paste a tweet here:")

if st.button("Predict Sentiment"):
    clean = text.lower()
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    if pred == 1:
        st.success("Positive 😀")
    else:
        st.error("Negative 😡")
