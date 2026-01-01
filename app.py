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

# ========================= BASE CONFIG ========================= #
st.set_page_config(
    page_title="IN Election Sentiment Analyzer",
    page_icon="🗳️",
    layout="wide"
)


# ========================= GLOBAL CSS ========================= #
st.markdown("""
<style>

/* Sidebar ALWAYS visible */
[data-testid="stSidebar"] {
    min-width: 240px !important;
    max-width: 240px !important;
    background: #f8f9fc !important;
    padding-top: 30px !important;
}

/******** MAIN CONTAINER ********/
.container {
    max-width: 900px;
    margin: auto;
    background: white;
    padding: 32px;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
}

/* Responsive container on mobile */
@media (max-width:600px){
    .container {
        max-width: 100% !important;
        padding: 20px !important;
    }
}

/******** TITLE ********/
h1.title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg,#0072ff,#00d97e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/******** FOOTER ********/
.footer-box {
    width: 100%;
    margin-top: 50px;
    padding: 20px;
    text-align: center;
    opacity: 0.8;
}
.footer-box a {
    text-decoration: none;
    margin: 0 12px;
    color: inherit;
}

/* Sidebar button highlight */
button[kind="secondary"] {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ========================= NAVIGATION ========================= #
if "page" not in st.session_state:
    st.session_state.page = "home"

def nav(label, key, icon):
    clicked = st.sidebar.button(f"{icon}  {label}", key=f"nav-{key}", use_container_width=True)
    if clicked:
        st.session_state.page = key
    # ACTIVE highlight
    if st.session_state.page == key:
        st.sidebar.markdown(
            f"""
            <style>
            button[kind="secondary"][key="nav-{key}"] {{
                background:#0072ff !important;
                color:white !important;
                border-color:#0072ff !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )


with st.sidebar:
    st.markdown("### 🌐 Navigation")
    nav("Home", "home", "🏠")
    nav("Analyze Tweet", "analyze", "📝")
    nav("Bulk CSV Analysis", "csv", "📁")
    nav("Dashboard", "dashboard", "📊")
    st.markdown("---")
    st.markdown("<p style='font-size:12px;text-align:center;opacity:0.6;'>Navigation</p>", unsafe_allow_html=True)


# ========================= UTILS ========================= #
def translate(text):
    try:
        if any("\u0900" <= ch <= "\u097F" for ch in text):  # Hindi Unicode range
            return GoogleTranslator(source="auto", target="en").translate(text)
        return text
    except:
        return text

def classify(text):
    text = translate(text).lower()
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    try:
        conf = max(model.predict_proba(vec)[0]) * 100
    except:
        conf = None
    return pred, conf

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
        if y < 40:
            c.showPage()
            y = 760
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# ========================= PAGES ========================= #
if st.session_state.page == "home":
    st.markdown('<h1 class="title">IN Election Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.write("<p style='text-align:center;'>Analyze tweets using NLP, ML & Hindi auto-translation.</p>", unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.subheader("✨ Features")
    st.write("""
    - 🧠 Tweet Sentiment Classification  
    - 📁 Upload CSV for Bulk Analysis  
    - 🌐 Hindi / multilingual auto-translation  
    - 📊 Dashboard metrics + pie chart  
    - 📎 CSV + PDF export report  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "analyze":
    st.markdown('<h1 class="title">Analyze Tweet</h1>', unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    text = st.text_area("✍️ Enter Tweet", height=160)
    if st.button("🔍 Predict Sentiment", use_container_width=True):
        if text.strip() == "":
            st.warning("⚠ Enter some text")
        else:
            pred, conf = classify(text)
            label = "😊 Positive" if pred == 1 else "😡 Negative"
            (st.success if pred == 1 else st.error)(f"{label} — Confidence: {conf:.2f}%")
            st.session_state.setdefault("history", []).append(
                {"Tweet": text, "Sentiment": label, "Confidence": f"{conf:.2f}%"})
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "csv":
    st.markdown('<h1 class="title">CSV Bulk Upload</h1>', unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV (must contain column 'text')", type="csv")
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
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "dashboard":
    st.markdown('<h1 class="title">Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    if "history" in st.session_state and len(st.session_state.history) > 0:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist)

        fig, ax = plt.subplots()
        counts = Counter(df_hist["Sentiment"])
        ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
        st.pyplot(fig)

        st.download_button("📎 Download CSV", df_hist.to_csv(index=False).encode("utf-8"), "sentiment_history.csv")
        st.download_button("📄 Download PDF", generate_pdf(df_hist), "report.pdf", "application/pdf")
    else:
        st.info("No history yet — analyze tweets first.")
    st.markdown("</div>", unsafe_allow_html=True)


# ========================= FOOTER ========================= #
st.markdown("""
<br>
<div class="footer-box">
🌐 <a href="https://rajlaljipandey.github.io/" target="_blank">Portfolio</a> •
🧠 <a href="https://github.com/rajlaljipandey" target="_blank">GitHub</a> •
✉️ <a href="mailto:rajlaljipandey@gmail.com" target="_blank">Contact</a>
<br><br>
Made with ❤️ by <b>Raj Lalji Pandey</b>
</div>
""", unsafe_allow_html=True)
