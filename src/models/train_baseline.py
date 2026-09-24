from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os


train = load_from_disk("src/data/processed/hc3/train")

X_train = train["text"]
y_train = train["label"]


vectorizer = TfidfVectorizer(
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)


model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

model.fit(X_train_tfidf, y_train)


os.makedirs("src/artifacts", exist_ok=True)

joblib.dump(
    model,
    "src/artifacts/model.pkl"
)

joblib.dump(
    vectorizer,
    "src/artifacts/vectorizer.pkl"
)

print("Model trained and saved successfully.")