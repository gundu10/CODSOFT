
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



data = []
with open("train_data.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ::: ")

        # Check if the line has all 4 fields
        if len(parts) == 4:
            movie_id = parts[0]
            title = parts[1]
            genre = parts[2]
            description = parts[3]

            data.append([movie_id, title, genre, description])

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

print("Dataset Loaded Successfully!")
print(df.head())



X = df["DESCRIPTION"]
y = df["GENRE"]



tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = tfidf.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy:", round(accuracy * 100, 2), "%")



joblib.dump(model, "genre_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")
print("Files Created:")
print("- genre_model.pkl")
print("- tfidf_vectorizer.pkl")