import streamlit as st
import pandas as pd
import spacy
import yake
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Load NLP models
@st.cache_resource
def load_models():
    nlp_spacy = spacy.load("en_core_web_sm")
    kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
    return nlp_spacy, kw_model

nlp_spacy, kw_model = load_models()

# Function to extract keywords using YAKE
def extract_yake(texts, max_keywords=5):
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=max_keywords)
    results = []
    for text in texts:
        keywords = kw_extractor.extract_keywords(text)
        results.append([kw[0] for kw in keywords])
    return results

# Function to extract keywords using spaCy noun chunks
def extract_spacy(texts):
    results = []
    for text in texts:
        doc = nlp_spacy(text)
        keywords = [chunk.text.lower() for chunk in doc.noun_chunks]
        results.append(keywords)
    return results

# Function to extract keywords using KeyBERT
def extract_keybert(texts, max_keywords=5):
    results = []
    for text in texts:
        keywords = kw_model.extract_keywords(text, top_n=max_keywords)
        results.append([kw[0] for kw in keywords])
    return results

# Streamlit UI
st.set_page_config(page_title="Keyword Extraction from Customer Feedback", layout="wide")
st.title("🔍 Keyword Extraction from Customer Feedback")

uploaded_file = st.file_uploader("Upload a CSV file with a `feedback` column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Sample Data", df.head())

    method = st.selectbox("Select Keyword Extraction Method", ["spaCy", "YAKE", "KeyBERT"])
    max_kw = st.slider("Number of Keywords per Feedback", 1, 10, 5)

    if st.button("Extract Keywords"):
        feedbacks = df["feedback"].dropna().astype(str).tolist()
        if method == "spaCy":
            keywords = extract_spacy(feedbacks)
        elif method == "YAKE":
            keywords = extract_yake(feedbacks, max_keywords=max_kw)
        else:
            keywords = extract_keybert(feedbacks, max_keywords=max_kw)

        df["keywords"] = keywords
        st.success("✅ Keywords Extracted!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="extracted_keywords.csv", mime="text/csv")
