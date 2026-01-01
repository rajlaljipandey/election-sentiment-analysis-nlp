import streamlit as st
import joblib
import pandas as pd
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO
from reportlab.pdfgen import canvas

# ========================= LOAD MODEL ========================= #
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ========================= CONFIG ========================= #
st.set_page_config(
    page_title="IN Election Sentiment Analyzer",
    page_icon="🗳️",
    layout="wide",
)

# Sidebar Navigation
menu = st.sidebar.radio("📌 Navigation", ["🏠 Home", "📝 Analyze Tweet", "📁 Upload CSV", "📡 Live Twitter", "📊 Dashboard"])

# Support Languages
SUPPORTED_LANGS = {
    "Hindi": "hi", "Gujarati": "gu", "Marathi": "mr",
    "Tamil": "ta", "Bengali": "bn", "English": "en"
}

# ========================= UTIL FUNCTIONS ========================= #
def translate(text, lang_code):
    try:
        if lang_code != "en":
            return GoogleTranslator(source="auto", target="en").translate(text)
        return text
    except:
        return text

def classify(text, lang_code="en"):
    text = translate(text, lang_code).lower()
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    try:
        confidence = max(model.predict_proba(vec)[0]) * 100
    except:
        confidence = None
    return pred, confidence

def generate_pdf(history_df):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 11)
    c.drawString(40, 800, "Sentiment Analysis Report")
    c.drawString(40, 785, "----------------------------")
    y = 760
    for _, row in history_df.iterrows():
        c.drawString(40, y, f"{row['Tweet'][:60]}...  → {row['Sentiment']} ({row['Confidence']})")
        y -= 18
        if y < 30:
            c.showPage()
            y = 760
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# ========================= GLOBAL CSS ========================= #
st.markdown("""
<style>
.container-box {
    max-width: 900px;
    margin: auto;
    background: white;
    padding: 32px;
    border-radius: 20px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.07);
}
h1.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg,#0072ff,#00d97e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ========================= MAIN PAGES ========================= #

# ----- HOME ----- #
if menu == "🏠 Home":
    st.markdown('<h1 class="title">IN Election Sentiment Analyzer 🇮🇳</h1>', unsafe_allow_html=True)
    st.write("Analyze Indian election tweets using NLP, Machine Learning, Live Twitter Scraping & Multi-Language Support.")
    st.markdown('<div class="container-box">', unsafe_allow_html=True)
    st.write("👈 Use the sidebar to explore features.")
    st.write("✨ Features Included:\n- Tweet Sentiment\n- CSV Bulk Analysis\n- Live Twitter Scraper\n- Dashboard Visuals\n- PDF & CSV Export\n- Hindi & Regional Language Translation")
    st.markdown('</div>', unsafe_allow_html=True)

# ----- ANALYZE TWEET ----- #
elif menu == "📝 Analyze Tweet":
    st.markdown('<h1 class="title">Tweet Sentiment Test</h1>', unsafe_allow_html=True)
    st.markdown('<div class="container-box">', unsafe_allow_html=True)

    tweet = st.text_area("🔥 Enter a Tweet", placeholder="Type or paste text...", height=160)
    lang = st.selectbox("🌐 Language:", list(SUPPORTED_LANGS.keys()))

    if st.button("🔍 Predict Sentiment", use_container_width=True):
        if tweet.strip() == "":
            st.warning("⚠ Please enter some text")
        else:
            pred, conf = classify(tweet, SUPPORTED_LANGS[lang])
            label = "😊 Positive" if pred == 1 else "😡 Negative"

            if pred == 1:
                st.success(f"{label} — Confidence: {conf:.2f}%")
            else:
                st.error(f"{label} — Confidence: {conf:.2f}%")

            st.session_state.setdefault("history", []).append(
                {"Tweet": tweet, "Sentiment": label, "Confidence": f"{conf:.2f}%"}
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ----- UPLOAD CSV BULK ----- #
elif menu == "📁 Upload CSV":
    st.markdown('<h1 class="title">Bulk CSV Upload</h1>', unsafe_allow_html=True)
    st.markdown('<div class="container-box">', unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV with column 'text'", type="csv")
    if file:
        df = pd.read_csv(file)
        results = []
        for text in df["text"]:
            pred, conf = classify(text)
            label = "Positive" if pred == 1 else "Negative"
            results.append([text, label, f"{conf:.2f}%"])
        new_df = pd.DataFrame(results, columns=["Tweet", "Sentiment", "Confidence"])
        st.dataframe(new_df, use_container_width=True)
        st.download_button("⬇ Download CSV", new_df.to_csv(index=False).encode("utf-8"), "bulk_output.csv")

    st.markdown('</div>', unsafe_allow_html=True)

# ----- LIVE TWITTER ----- #
elif menu == "📡 Live Twitter":
    st.markdown('<h1 class="title">Real-Time Twitter Scraper</h1>', unsafe_allow_html=True)
    st.markdown('<div class="container-box">', unsafe_allow_html=True)

    query = st.text_input("Enter keyword (#Election2024, Modi, Congress)")
    count = st.slider("Tweets to fetch", 5, 100, 10)

    if st.button("📡 Fetch Tweets", use_container_width=True):
        tweets = []
        for i, tweet_obj in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= count: break
            pred, conf = classify(tweet_obj.content)
            sentiment = "Positive" if pred == 1 else "Negative"
            tweets.append([tweet_obj.date, tweet_obj.username, tweet_obj.content, sentiment, f"{conf:.2f}%"])
        live_df = pd.DataFrame(tweets, columns=["Date", "User", "Tweet", "Sentiment", "Confidence"])
        st.dataframe(live_df, use_container_width=True)
        st.download_button("⬇ Save CSV", live_df.to_csv(index=False).encode("utf-8"), "live_tweets.csv")

    st.markdown('</div>', unsafe_allow_html=True)

# ----- DASHBOARD ----- #
elif menu == "📊 Dashboard":
    st.markdown('<h1 class="title">Sentiment Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="container-box">', unsafe_allow_html=True)

    if "history" in st.session_state and len(st.session_state.history) > 0:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist)

        fig, ax = plt.subplots()
        counts = Counter(df_hist["Sentiment"])
        ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
        st.pyplot(fig)

        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv, "sentiment_history.csv")

        pdf = generate_pdf(df_hist)
        st.download_button("📄 Download PDF", pdf, "report.pdf", "application/pdf")
    else:
        st.info("No history yet — analyze tweets first.")

    st.markdown('</div>', unsafe_allow_html=True)

# ----- FOOTER ----- #
st.markdown('<p class="footer" style="text-align:center;">Built ❤️ by Raj Lalji Pandey</p>', unsafe_allow_html=True)
