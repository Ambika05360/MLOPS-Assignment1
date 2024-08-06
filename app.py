from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor
model = joblib.load("20240801_163255_best_model.joblib")
preprocessor = joblib.load("preprocessor.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    processed_data = preprocessor.transform(df)
    predictions = model.predict(processed_data)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
