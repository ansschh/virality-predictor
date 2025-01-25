# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
from typing import List
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and scaler at startup
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model = tf.keras.models.load_model('trained_model.keras')
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

class ContentInput(BaseModel):
    text: str
    title: str
    language: str
    post_hour: int
    post_day: int
    is_weekend: int

class PredictionResponse(BaseModel):
    virality_score: float
    processed_features: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict_virality(content: ContentInput):
    try:
        # Create feature vector
        features = {
            'text_length': len(content.text),
            'title_length': len(content.title),
            'language': content.language,
            'post_hour': content.post_hour,
            'post_day': content.post_day,
            'is_weekend': content.is_weekend
        }
        
        # Convert to numpy array and reshape
        feature_vector = np.array([list(features.values())])
        
        # Scale features
        scaled_features = scaler.transform(feature_vector)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Convert from log space
        virality_score = float(np.expm1(prediction[0][0]))
        
        return PredictionResponse(
            virality_score=virality_score,
            processed_features=features
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)