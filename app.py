from flask import Flask, jsonify, request
import joblib
import os
import numpy as np
from pyngrok import ngrok

app = Flask(__name__)

# Load the latest model
def load_latest_model():
    model_files = [f for f in os.listdir() if f.startswith('GS_model') and f.endswith('.joblib')]
    if not model_files:
        raise RuntimeError("No model files found")
    latest_model_file = max(model_files, key=os.path.getctime)
    return joblib.load(latest_model_file)

model = load_latest_model()

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask app!"})

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the request contains JSON data
    if not request.json:
        return jsonify({"error": "No input data provided"}), 400
    
    # Extract features from the request data
    input_data = request.json.get('features')
    if input_data is None:
        return jsonify({"error": "No 'features' key in input data"}), 400
    
    try:
        # Convert input data to numpy array and make prediction
        input_array = np.array(input_data).reshape(1, -1)  # Reshape for single sample prediction
        prediction = model.predict(input_array)
        # Return the prediction result
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start Flask app
    port = 5000
    # Start ngrok tunnel
    ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # Optional, if you have an ngrok auth token
    public_url = ngrok.connect(port)
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    app.run(host='0.0.0.0', port=port)
