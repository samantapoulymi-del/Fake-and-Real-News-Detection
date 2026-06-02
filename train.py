import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

fake = pd.read_csv("Fake_news.csv")
true = pd.read_csv("Real_news.csv")

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true], ignore_index=True)

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

X = df["content"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "fake_news_model.pkl")

print("Model saved!")