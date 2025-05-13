from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os
app = Flask(__name__)
CORS(app)


# ▶️ Load model and vectorizer
model_path      = os.path.join(os.path.dirname(__file__), "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

print(f"🔄 Loading model from {model_path}")
model      = pickle.load(open(model_path,      "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))
print("✅ Model and vectorizer loaded")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # vectorize & predict
    vec        = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    print(f"🔍 Predicted '{prediction}' for input: {text[:30]}…")
    return jsonify({"result": prediction})

if __name__ == "__main__":
    # ▶️ This starts the server and blocks the terminal
    app.run(host="127.0.0.1", port=5000, debug=True)
