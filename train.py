import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("Loading Dataset...")

df = pd.read_csv("spam (1).csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("\nDataset Loaded Successfully!")
print(df.head())

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

X = df['message']
y = df['label']


tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X = tfidf.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Naive Bayes Model...")

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

pred_nb = nb_model.predict(X_test)

print("\nNaive Bayes Accuracy:",
      round(accuracy_score(y_test, pred_nb) * 100, 2), "%")

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

pred_lr = lr_model.predict(X_test)

print("\nLogistic Regression Accuracy:",
      round(accuracy_score(y_test, pred_lr) * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, pred_lr))

joblib.dump(nb_model, "spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")
print("Created:")
print("- spam_model.pkl")
print("- tfidf_vectorizer.pkl")