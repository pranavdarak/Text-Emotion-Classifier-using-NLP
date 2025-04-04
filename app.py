from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

# Load model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Emoji dict
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", 
    "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form['text']
        prediction = pipe_lr.predict([text])[0]
        probabilities = pipe_lr.predict_proba([text])[0]
        confidence = np.max(probabilities)

        proba_df = pd.DataFrame({
            "Emotion": pipe_lr.classes_,
            "Probability": probabilities
        })

        chart_data = proba_df.sort_values(by="Probability", ascending=False).to_dict(orient="records")
        return render_template("index.html", 
                               text=text,
                               prediction=prediction,
                               emoji=emotions_emoji_dict.get(prediction, "❓"),
                               confidence=round(confidence * 100, 2),
                               chart_data=json.dumps(chart_data))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)