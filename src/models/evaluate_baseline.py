from datasets import load_from_disk
from sklearn.metrics import accuracy_score, classification_report
import joblib


validation = load_from_disk("src/data/processed/hc3/validation")

X_validation = validation["text"]
y_validation = validation["label"]

model = joblib.load("src/artifacts/model.pkl")
vectorizer = joblib.load("src/artifacts/vectorizer.pkl")

X_validation_tfidf = vectorizer.transform(X_validation)
predictions = model.predict(X_validation_tfidf)
accuracy = accuracy_score(y_validation,predictions)

print("Validation Accuracy:", accuracy)
print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        predictions,
        target_names=["Human", "AI"]
    )
)