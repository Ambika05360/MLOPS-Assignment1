import os
import joblib
import logging
from flask import Flask, jsonify, request

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_latest_model():
    model_files = [f for f in os.listdir() if f.startswith('GS_model') and f.endswith('.joblib')]
    if not model_files:
        logging.error("No model files found")
        raise RuntimeError("No model files found")
    latest_model_file = max(model_files, key=os.path.getctime)
    logging.info(f"Loading model from file: {latest_model_file}")
    return joblib.load(latest_model_file)

try:
    model = load_latest_model()
except RuntimeError as e:
    logging.error(f"Model loading failed: {e}")
    model = None

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask app!"})

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({"error": "No input data provided"}), 400

    input_data = request.json.get('features')
    if input_data is None:
        return jsonify({"error": "No 'features' key in input data"}), 400

    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
