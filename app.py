from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Performance metrics
accuracy = "98.45%"
precision = "98.00%"
recall = "99.00%"
f1_score = "98.00%"

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    news = ""

    if request.method == "POST":

        news = request.form["news"]

        # Convert text into vector
        transformed_news = vectorizer.transform([news])

        # Predict
        result = model.predict(transformed_news)[0]

        # Confidence score
        confidence_score = model.predict_proba(transformed_news)[0]

        if result == 1:
            prediction = "🟢 REAL NEWS"
            confidence = round(confidence_score[1] * 100, 2)
        else:
            prediction = "🔴 FAKE NEWS"
            confidence = round(confidence_score[0] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news=news,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)