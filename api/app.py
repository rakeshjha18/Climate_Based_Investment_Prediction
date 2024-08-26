from flask import Flask, request, jsonify
from tensorflow.python.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("../models/lstm_model.h5")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(data)
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
