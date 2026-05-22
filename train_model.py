import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle
data = data.sample(frac=1, random_state=42)

# Use title + text
data["content"] = data["title"] + " " + data["text"]

# Features and labels
X = data["content"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

Xv_train = vectorizer.fit_transform(X_train)
Xv_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()

model.fit(Xv_train, y_train)

# Predictions
pred = model.predict(Xv_test)

# Metrics
print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and Vectorizer saved successfully!")