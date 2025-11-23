from pathlib import Path
import joblib
import numpy as np

# Base directory = project root (SmartPitch-ML-Controller)
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models_wind_profile" / "rf_wind_profile_model.pkl"
SCALER_PATH = BASE_DIR / "models_wind_profile" / "scaler_wind_profile.pkl"

# Load model and scaler once when the API starts
rf_wind_profile_model = joblib.load(MODEL_PATH)
wind_profile_scaler = joblib.load(SCALER_PATH)


def predict_wind_profile_pitch(hor_windv: float, rot_speed: float, gen_pwr: float) -> float:
    """
    Predict Region 3 blade pitch using the wind-profile Random Forest model.

    Inputs:
        hor_windv  [m/s]  - horizontal wind speed
        rot_speed  [rpm]  - rotor speed
        gen_pwr    [kW]   - generator power
    """
    # Create 2D array: shape (1, 3)
    x = np.array([[hor_windv, rot_speed, gen_pwr]])

    # Scale inputs
    x_scaled = wind_profile_scaler.transform(x)

    # Predict pitch
    pitch_pred = rf_wind_profile_model.predict(x_scaled)[0]

    # Return as normal float
    return float(pitch_pred)
