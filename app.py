from flask import Flask, jsonify, request
import joblib
import os
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the latest model
def load_latest_model():
    try:
        model_files = [f for f in os.listdir() if f.endswith('.joblib')]
        if not model_files:
            raise RuntimeError("No model files found")
        latest_model_file = max(model_files, key=os.path.getctime)
        app.logger.info(f"Loading model: {latest_model_file}")
        return joblib.load(latest_model_file)
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        raise

model = load_latest_model()

@app.route('/')
def index():
    app.logger.info("Index route called")
    return jsonify({"message": "Welcome to the Flask app!"})

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Predict route called")
    if not request.json:
        app.logger.error("No input data provided")
        return jsonify({"error": "No input data provided"}), 400

    input_data = request.json.get('features')
    if input_data is None:
        app.logger.error("No 'features' key in input data")
        return jsonify({"error": "No 'features' key in input data"}), 400

    try:
        columns = ['year', 'gender', 'age', 'location', 'race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'hbA1c_level', 'blood_glucose_level']
        input_df = pd.DataFrame([input_data], columns=columns)
        input_df = input_df.reindex(columns=columns, fill_value=0)  # Ensure all columns are present
        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
