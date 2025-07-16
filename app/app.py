import streamlit as st
import joblib
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import clean_text

# Load model & vectorizer
model = joblib.load("models/news_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Category emojis
category_emojis = {
    'World': '🌍',
    'Politics': '🏛️',
    'Business': '💼',
    'Tech': '💻',
    'Sports': '🏅',
    'Entertainment': '🎬'
}

# App title
st.set_page_config(page_title="🗞️ News Category Classifier", layout="centered")
st.title("🗞️ News Category Classifier")
st.markdown("Predict the category of a news headline using an ML model trained on real-world data.")

# Text input
headline = st.text_area("✏️ Enter a news headline:", height=100)

if st.button("🔍 Classify"):
    if headline.strip() == "":
        st.warning("Please enter a headline to classify.")
    else:
        cleaned = clean_text(headline)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        st.success(f"Predicted Category: **{prediction}**")
