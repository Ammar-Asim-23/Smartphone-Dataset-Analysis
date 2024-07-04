from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)
# Load your pre-trained model
model = joblib.load('best_rf_model.pb')

scaler = joblib.load('scaler.pb')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.json
    # Extract input features from the form data
    resolution_height = float(data['resolution_height'])
    processor_speed = float(data['processor_speed'])
    screen_size = float(data['screen_size'])
    internal_memory = float(data['internal_memory'])
    resolution_width = float(data['resolution_width'])
    primary_camera_front = float(data['primary_camera_front'])
    rating = float(data['rating'])
    ram_capacity = float(data['ram_capacity'])
    has_nfc = int(data['has_nfc'])
    extended_memory_available = int(data['extended_memory_available'])
    primary_camera_rear = float(data['primary_camera_rear'])
    os_ios = int(data['os_ios'])
    refresh_rate = float(data['refresh_rate'])
    has_5g = int(data['has_5g'])
    num_rear_cameras = int(data['num_rear_cameras'])
    num_cores = int(data['num_cores'])
    processor_brand_snapdragon = int(data['processor_brand_snapdragon'])

    # Create input features array
    features = np.array([[
        resolution_height, processor_speed, screen_size, internal_memory,
        resolution_width, primary_camera_front, rating, ram_capacity,
        has_nfc, extended_memory_available, primary_camera_rear, os_ios,
        refresh_rate, has_5g, num_rear_cameras, num_cores, processor_brand_snapdragon
    ]])

    # Scale the input features using the loaded scaler
    scaled_features = scaler.transform(features)

    # Make prediction using the model
    predicted_price = model.predict(scaled_features)[0]

    # Return the predicted price as JSON response
    return jsonify({'predicted_price': round(predicted_price,-2)})

if __name__ == '__main__':
    app.run(debug=True)






