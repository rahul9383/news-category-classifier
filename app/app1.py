from src.predict import predict_category

user_input = input("Enter a News Headline: ")

category = predict_category(user_input)
print(f"Predicted Category: {category}")
