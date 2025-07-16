import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import streamlit as st
import joblib
from src.preprocess import clean_text

st.title("📰 News Category Classifier")

try:
    st.write("⏳ Loading model...")
    model = joblib.load("models/news_classifier.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    st.write("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

user_input = st.text_area("Enter a News Headline:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a headline.")
    else:
        try:
            cleaned = clean_text(user_input)
            vect = vectorizer.transform([cleaned])
            prediction = model.predict(vect)[0]
            st.success(f"📣 Predicted Category: **{prediction}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
