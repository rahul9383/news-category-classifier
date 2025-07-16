import joblib
from src.preprocess import clean_text

model = joblib.load("models/news_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_category(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return prediction