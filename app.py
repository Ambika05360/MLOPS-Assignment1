from flask import Flask, jsonify
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction Model API!"

@app.route('/predict', methods=['GET'])
def predict():
    # Load the trained model
    model = joblib.load('GS_model.joblib')

    # For simplicity, let's use dummy data for prediction
    dummy_data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

    # Predict using the loaded model
    prediction = model.predict(dummy_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
