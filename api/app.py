"""
Production-ready FastAPI application for fake news detection.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from typing import Optional
from src.preprocessing import clean_text, extract_text_features

app = FastAPI(title="Fake News Detection API")

# Pydantic schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top_features: Optional[list] = None

# Load model and vectorizer
MODEL_PATH = os.path.join("models", "fake_news_model_logistic.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
SCALER_PATH = os.path.join("models", "feature_scaler.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

@app.get("/")
def health_check():
    return {"message": "Fake News Detection API is live"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Preprocess input
    cleaned = clean_text(request.text, use_advanced_features=True, preserve_entities=True)
    features = extract_text_features(cleaned, use_advanced_features=True)
    # Vectorize
    X_tfidf = vectorizer.transform([cleaned])
    # Combine with features if scaler exists
    if scaler and features:
        import numpy as np
        import pandas as pd
        feature_df = pd.DataFrame([features])
        X_features = scaler.transform(feature_df)
        X_combined = np.hstack([X_tfidf.toarray(), X_features])
    else:
        X_combined = X_tfidf.toarray()
    # Predict
    pred = model.predict(X_combined)[0]
    proba = model.predict_proba(X_combined)[0]
    confidence = float(max(proba))
    label = "fake" if pred == 1 else "real"
    # Optionally, get top features (not implemented here)
    return PredictResponse(prediction=label, confidence=confidence) 