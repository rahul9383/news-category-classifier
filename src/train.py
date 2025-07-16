# src/train.py
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from src.preprocess import clean_text

print("🚀 Starting training...")

# Step 1: Load dataset
data_path = "data/News_Category_Dataset_v3.json"
if not os.path.exists(data_path):
    print(f"❌ Dataset not found at: {data_path}")
    exit()

print("📥 Loading dataset...")
with open(data_path, 'r') as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

print(f"✅ Loaded {len(df)} rows")

# Step 2: Preprocess
print("🧼 Cleaning text...")
df['text'] = df['headline'].apply(clean_text)
X = df['text']
y = df['category']

# Step 3: Split
print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize
print("🔠 Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train model
print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Step 6: Evaluate
print("📊 Evaluating model...")
preds = model.predict(X_test_vec)
report = classification_report(y_test, preds)
print(report)

# Step 7: Save model
print("💾 Saving model and vectorizer...")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/news_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Done! Model saved to 'models/'")
