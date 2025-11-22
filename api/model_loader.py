import joblib
import os

# Path to /api/models/
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Correct model + correct scaler
SCALER_PATH = os.path.join(MODEL_DIR, "standard_scaler.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_pitch_regressor_best.joblib")

# ---- Physical pitch limits (deg) ----
PITCH_MIN = 0.0
PITCH_MAX = 30.0

# Load once at startup
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)


def predict_pitch_with_raw(wind_speed: float, rotor_speed: float, power: float):
    """
    Return BOTH:
      - clipped_pitch : saturated to [PITCH_MIN, PITCH_MAX]
      - raw_pitch     : unconstrained model output
    """
    X = [[wind_speed, rotor_speed, power]]
    X_scaled = scaler.transform(X)

    raw_pitch = float(model.predict(X_scaled)[0])
    clipped_pitch = max(PITCH_MIN, min(PITCH_MAX, raw_pitch))

    if clipped_pitch != raw_pitch:
        print(
            f"[WARN] Pitch prediction clipped from {raw_pitch:.2f}° "
            f"to {clipped_pitch:.2f}°"
        )

    return clipped_pitch, raw_pitch


def predict_pitch(wind_speed: float, rotor_speed: float, power: float) -> float:
    """
    Backwards-compatible helper that returns only the clipped pitch.
    (Kept in case anything else still imports predict_pitch.)
    """
    clipped, _ = predict_pitch_with_raw(wind_speed, rotor_speed, power)
    return clipped
