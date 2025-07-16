# test_app.py
import streamlit as st

st.set_page_config(page_title="Streamlit Test", layout="centered")
st.title("✅ Streamlit is Working")

name = st.text_input("What's your name?")
if name:
    st.success(f"Hello, {name}! Streamlit is working perfectly.")
