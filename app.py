from flask import Flask, jsonify, request
import joblib
import os
import numpy as np

app = Flask(__name__)

def load_latest_model():
    try:
        model_files = [f for f in os.listdir() if f.startswith('GS_model') and f.endswith('.joblib')]
        if not model_files:
            raise RuntimeError("No model files found")
        latest_model_file = max(model_files, key=os.path.getctime)
        print(f"Loading model file: {latest_model_file}")  # Log model file name
        model = joblib.load(latest_model_file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_latest_model()

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask app!"})

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the request contains JSON data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Extract features from the request data
    data = request.get_json()
    input_data = data.get('features')
    if input_data is None:
        return jsonify({"error": "No 'features' key in input data"}), 400

    try:
        print(f"Received input data: {input_data}")  # Log input data
        input_array = np.array(input_data).reshape(1, -1)  # Reshape for single sample prediction
        print(f"Input array shape: {input_array.shape}")  # Log input array shape
        prediction = model.predict(input_array)
        print(f"Prediction result: {prediction}")  # Log prediction result
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log any exception
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run Flask app with debug mode enabled
    app.run(host='0.0.0.0', port=5000, debug=True)
