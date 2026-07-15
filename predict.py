
import joblib

model = joblib.load("genre_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

print("Movie Genre Prediction")
print("-" * 40)

while True:
    description = input("\nEnter Movie Description:\n")
    description_tfidf = tfidf.transform([description])
    prediction = model.predict(description_tfidf)

    print("\nPredicted Genre:", prediction[0])

    choice = input("\nPredict another movie? (yes/no): ").lower()

    if choice != "yes":
        print("\nThank you!")
        break