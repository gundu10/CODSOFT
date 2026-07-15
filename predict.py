import joblib

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

print("======================================")
print("      SMS Spam Detection System")
print("======================================")

while True:
    message = input("\nEnter an SMS message:\n")

    message_vector = tfidf.transform([message])

    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        print("\nPrediction: SPAM")
    else:
        print("\nPrediction: HAM (Legitimate Message)")

    choice = input("\nDo you want to test another message? (yes/no): ").lower()

    if choice != "yes":
        print("\nThank you!")
        break