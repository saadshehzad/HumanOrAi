from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer


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