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
    Predict Region 3 blade pitch using the wind-profile Random Forest model
    for a SINGLE operating point.

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


# ========= NEW: time-series prediction =========

def predict_wind_profile_series(
    ws_list: list[float],
    rs_list: list[float],
    gp_list: list[float],
) -> list[float]:
    """
    Predict Region 3 blade pitch for a FULL TIME SERIES.

    All lists must have the same length N:

        ws_list[i]  = wind speed at time i
        rs_list[i]  = rotor speed at time i
        gp_list[i]  = generator power at time i

    Returns:
        list of length N with pitch angles [deg].
    """
    n_ws = len(ws_list)
    n_rs = len(rs_list)
    n_gp = len(gp_list)

    if n_ws == 0:
        raise ValueError("wind_speeds list must not be empty")
    if not (n_ws == n_rs == n_gp):
        raise ValueError("wind_speeds, rotor_speeds, gen_powers must have SAME length")

    # Build feature matrix X: shape (N, 3)
    X = np.column_stack([ws_list, rs_list, gp_list])

    # Scale inputs
    X_scaled = wind_profile_scaler.transform(X)

    # Predict pitch for each row
    pitch_preds = rf_wind_profile_model.predict(X_scaled)

    # Return as Python list
    return pitch_preds.tolist()
