import streamlit as st
import pandas as pd
import subprocess
import spacy
import yake
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Load NLP models with auto-download fallback
@st.cache_resource
def load_models():
    try:
        nlp_spacy = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp_spacy = spacy.load("en_core_web_sm")

    kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
    return nlp_spacy, kw_model

nlp_spacy, kw_model = load_models()

# Keyword extraction methods
def extract_yake(texts, max_keywords=5):
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=max_keywords)
    return [[kw[0] for kw in kw_extractor.extract_keywords(text)] for text in texts]

def extract_spacy(texts):
    return [[chunk.text.lower() for chunk in nlp_spacy(text).noun_chunks] for text in texts]

def extract_keybert(texts, max_keywords=5):
    return [[kw[0] for kw in kw_model.extract_keywords(text, top_n=max_keywords)] for text in texts]

# Streamlit UI
st.set_page_config(page_title="Keyword Extraction from Customer Feedback", layout="wide")
st.title("🔍 Keyword Extraction from Customer Feedback")

uploaded_file = st.file_uploader("📂 Upload a CSV file with a `feedback` column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "feedback" not in df.columns:
        st.error("❌ CSV must contain a column named 'feedback'.")
    else:
        st.write("📋 Sample Data", df.head())

        method = st.selectbox("📌 Select Keyword Extraction Method", ["spaCy", "YAKE", "KeyBERT"])
        max_kw = st.slider("🔢 Number of Keywords per Feedback", 1, 10, 5)

        if st.button("🚀 Extract Keywords"):
            feedbacks = df["feedback"].dropna().astype(str).tolist()

            with st.spinner("Processing..."):
                if method == "spaCy":
                    keywords = extract_spacy(feedbacks)
                elif method == "YAKE":
                    keywords = extract_yake(feedbacks, max_keywords=max_kw)
                else:
                    keywords = extract_keybert(feedbacks, max_keywords=max_kw)

            df["keywords"] = keywords
            st.success("✅ Keywords extracted!")
            st.dataframe(df.head(20))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results as CSV", data=csv, file_name="extracted_keywords.csv", mime="text/csv")
