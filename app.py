from flask import Flask, request, jsonify
import joblib
 
app = Flask(__name__)
 
# Load the model
model = joblib.load('m3-model.pkl')
 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data)
        features = data.get('features')
        if features is None:
            return jsonify({'error': 'No features provided'}), 400
 
        # Make prediction
        prediction = model.predict([features])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
if __name__ == '__main__':
    app.run(debug=True)
 
#curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\": [1.5, 2.5, 3.5, 4.5]}"
