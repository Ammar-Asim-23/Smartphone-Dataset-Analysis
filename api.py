from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('best_rf_model.pb')

scaler = joblib.load('scaler.pb')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    # Extract input features from the form data
    resolution_height = float(request.form['resolution_height'])
    processor_speed = float(request.form['processor_speed'])
    screen_size = float(request.form['screen_size'])
    internal_memory = float(request.form['internal_memory'])
    resolution_width = float(request.form['resolution_width'])
    primary_camera_front = float(request.form['primary_camera_front'])
    rating = float(request.form['rating'])
    ram_capacity = float(request.form['ram_capacity'])
    has_nfc = int(request.form['has_nfc'])
    extended_memory_available = int(request.form['extended_memory_available'])
    primary_camera_rear = float(request.form['primary_camera_rear'])
    os_ios = int(request.form['os_ios'])
    refresh_rate = float(request.form['refresh_rate'])
    has_5g = int(request.form['has_5g'])
    num_rear_cameras = int(request.form['num_rear_cameras'])
    num_cores = int(request.form['num_cores'])
    processor_brand_snapdragon = int(request.form['processor_brand_snapdragon'])

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






