from flask import Flask, jsonify, request
import joblib
import os

app = Flask(__name__)

# Load the latest model
def load_latest_model():
    # Get a list of all joblib files in the current directory
    model_files = [f for f in os.listdir() if f.startswith('GS_model') and f.endswith('.joblib')]
    
    # Ensure there are model files present
    if not model_files:
        raise FileNotFoundError("No model files found")
    
    # Find the latest model file by creation time
    latest_model_file = max(model_files, key=os.path.getctime)
    return joblib.load(latest_model_file)

# Load the model when the application starts
try:
    model = load_latest_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Ensure model is set to None if loading fails

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask app!"})

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the request contains JSON data
    if not request.json:
        return jsonify({"error": "No input data provided"}), 400
    
    # Extract features from the request data
    input_data = request.json
    
    # Check if the model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Implement prediction logic here
    try:
        # Assuming input_data is suitable for model prediction
        prediction = model.predict([input_data])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure the app listens on all interfaces and the specified port
    app.run(host='0.0.0.0', port=5000)
