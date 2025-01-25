# app.py

import joblib
from flask_cors import CORS
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS  # Import flask-cors

app = Flask(__name__)
CORS(app)

# Load model and scaler at startup
model = load_model('trained_model.keras')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Welcome to the Virality Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON input with the same features used during training.
    Example JSON payload:
    {
        "text_length": 500,
        "title_length": 50,
        "language": 1,
        "post_hour": 14,
        "post_day": 2,
        "is_weekend": 0,
        "total_views": 100,
        "total_likes": 5,
        "total_comments": 2,
        "total_follows": 1,
        "total_bookmarks": 0,
        "unique_users": 80,
        "unique_countries": 3,
        "engagement_duration": 50000
    }
    """
    data = request.get_json()
    
    # Put the input into the correct order (as used in training)
    # Make sure the features here match exactly the order in X
    # For example:
    features = [
        data["text_length"],
        data["title_length"],
        data["language"],
        data["post_hour"],
        data["post_day"],
        data["is_weekend"],
        data["total_views"],
        data["total_likes"],
        data["total_comments"],
        data["total_follows"],
        data["total_bookmarks"],
        data["unique_users"],
        data["unique_countries"],
        data["engagement_duration"]
    ]
    
    # Convert to dataframe or 2D array for the scaler
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    # Predict using the trained model
    prediction_log_space = model.predict(features_scaled)[0][0]
    
    # Convert from log space (since we used np.log1p during training)
    prediction = np.expm1(prediction_log_space)
    
    return jsonify({"prediction": float(prediction)})

if __name__ == "__main__":
    # Use the port provided by Render via the $PORT environment variable
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 for local testing
    app.run(host="0.0.0.0", port=port)