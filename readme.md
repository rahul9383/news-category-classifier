This project classifies news headlines into categories using a Logistic Regression model trained on the News Category Dataset.

## 🔧 Features
- Text preprocessing with stopword removal
- TF-IDF vectorization
- Logistic Regression classifier
- Streamlit web app for user interaction

## 📦 How to Run
1. Clone this repo and download the dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python src/train.py
   ```
4. Run the app:
   ```bash
   streamlit run app/app.py
   ```

## 📁 Dataset
- Place `News_Category_Dataset_v2.json` inside the `data/` folder.

## ✅ Sample Categories
- POLITICS, WELLNESS, ENTERTAINMENT, TRAVEL, STYLE & BEAUTY, PARENTING, HEALTH, etc.

---
Now you're ready to classify headlines in real-time!
