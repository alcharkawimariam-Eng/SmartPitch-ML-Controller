from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Region 3 Pitch Controller API")

MODEL_PATH = "models/ml_pitch_controller.joblib"
SCALER_PATH = "models/feature_scaler.joblib"

class PitchRequest(BaseModel):
    wind_speed: float
    rotor_speed: float
    generator_speed: float
    power: float
    pitch_prev: float

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None

@app.get("/")
def root():
    return {"message": "Region 3 Pitch Controller API is running."}

@app.post("/predict_pitch")
def predict_pitch(req: PitchRequest):
    if model is None:
        return {"error": "Model not loaded. Train and save your model to models/ml_pitch_controller.joblib"}
    if scaler is None:
        return {"error": "Scaler not loaded. Save a fitted scaler to models/feature_scaler.joblib"}
    x = np.array([[
        req.wind_speed,
        req.rotor_speed,
        req.generator_speed,
        req.power,
        req.pitch_prev
    ]])
    x_scaled = scaler.transform(x)
    pred = float(model.predict(x_scaled)[0])
    return {"predicted_pitch": pred}
