import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
label_encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')
model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

def preprocess_input(city, date_str):
    date = pd.to_datetime(date_str)

    city_encoded = label_encoder.transform([city])[0]

    input_data = pd.DataFrame({
        'City': [city_encoded],
        'Year': [date.year],
        'Month': [date.month],
        'Day': [date.day]
    })

    return input_data

@app.route('/predict_temperature', methods=['POST'])
def predict_temperature():
    try:
       
        data = request.json
        if 'city' not in data or 'date' not in data:
            return jsonify({'error': 'Missing city or date'}), 400

        city = data['city']
        date_str = data['date']
        input_data = preprocess_input(city, date_str)
        prediction = model.predict(input_data)[0]
        return jsonify({'temperature': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
