from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


train = load_from_disk("src/data/processed/hc3/train")
validation = load_from_disk("src/data/processed/hc3/validation")

print("Train:", len(train))
print("Validation:", len(validation))


X_train = train["text"]
y_train = train["label"]

X_validation = validation["text"]
y_validation = validation["label"]


vectorizer = TfidfVectorizer(
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_validation_tfidf = vectorizer.transform(X_validation)

print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Validation TF-IDF shape:", X_validation_tfidf.shape)


model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

validation_predictions = model.predict(X_validation_tfidf)

print("Validation predictions:", validation_predictions[:10])


accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

print("Validation Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        validation_predictions,
        target_names=["Human", "AI"]
    )
)