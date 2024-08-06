from flask import Flask, jsonify, request
import joblib
import os

app = Flask(__name__)

# Load the latest model
model_files = [f for f in os.listdir() if f.startswith('GS_model') and f.endswith('.joblib')]
latest_model_file = max(model_files, key=os.path.getctime)
model = joblib.load(latest_model_file)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask app!"})

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({"error": "No input data provided"}), 400

    input_data = request.json
    # Implement prediction logic here
    return jsonify({"message": "Prediction endpoint"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
