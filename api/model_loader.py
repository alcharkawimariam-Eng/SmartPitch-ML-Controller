import joblib
import os

# Path to /api/models/
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Correct model + correct scaler
SCALER_PATH = os.path.join(MODEL_DIR, "standard_scaler.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_pitch_regressor_best.joblib")

# Load once at startup
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

def predict_pitch(wind_speed, rotor_speed, power):
    X = [[wind_speed, rotor_speed, power]]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return float(pred)
