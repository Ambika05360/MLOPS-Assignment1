from flask import Flask, jsonify, request
import joblib
import os
from datetime import datetime

app = Flask(__name__)

def load_model():
    try:
        model_files = [f for f in os.listdir() if f.startswith('GS_model') and f.endswith('.joblib')]
        if not model_files:
            raise FileNotFoundError("No model file found")
        latest_model_file = max(model_files, key=os.path.getctime)
        model = joblib.load(latest_model_file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_model()

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask app!"})

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({"error": "No input data provided"}), 400

    input_data = request.json

    # Ensure input_data is in the correct format for the model
    # For example, you might need to convert it to a format that the model expects
    try:
        # Example: prediction = model.predict([input_data])
        prediction = "placeholder"  # Replace this with actual prediction logic
        return jsonify({"prediction": prediction})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Error during prediction"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
