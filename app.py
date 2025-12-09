from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import keras
from utils import preprocess_image

app = Flask(__name__)

# Load models
cnn = keras.models.load_model("models/cnn_model.keras")
rf = joblib.load("models/random_forest_model.joblib")
logreg = joblib.load("models/logistic_regression_model.joblib")

labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img, preview_img = preprocess_image(file)

    # ----- CNN -----
    cnn_probs = cnn.predict(img)[0] # type: ignore
    cnn_pred = labels[np.argmax(cnn_probs)]

    # Flatten for classical models
    flat = img.reshape(1, -1)

    # ----- Random Forest -----
    rf_probs = rf.predict_proba(flat)[0]
    rf_pred = labels[np.argmax(rf_probs)]

    # ----- Logistic Regression -----
    log_probs = logreg.predict_proba(flat)[0]
    log_pred = labels[np.argmax(log_probs)]

    return jsonify({
        "cnn_pred": cnn_pred,
        "cnn_probs": cnn_probs.tolist(),
        "rf_pred": rf_pred,
        "rf_probs": rf_probs.tolist(),
        "log_pred": log_pred,
        "log_probs": log_probs.tolist(),
        "labels": labels,
        "preview_img": preview_img
    })

if __name__ == "__main__":
    app.run(debug=True)
