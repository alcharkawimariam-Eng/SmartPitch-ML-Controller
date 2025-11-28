from pydantic import BaseModel
from typing import Optional, List


# ========= Pitch models (Region 3 MLP) =========

class PitchRequest(BaseModel):
    wind_speed: float
    rotor_speed: float
    power: float


class PitchResponse(BaseModel):
    # clipped (safe) command
    pitch: float
    # unconstrained model output (may be >30°)
    pitch_raw: Optional[float] = None


# Batch versions
class PitchBatchRequest(BaseModel):
    samples: List[PitchRequest]


class PitchBatchResponse(BaseModel):
    results: List[PitchResponse]


# ========= Wind-profile models (Random Forest) =========
# ---- Single-point version (ALREADY WORKING, KEEP IT) ----

class WindProfileRequest(BaseModel):
    """
    Request body for /predict_wind_profile.

    Matches what Streamlit single-point tab sends:
    {
      "wind_speeds": [0],
      "rotor_speed": 0,
      "gen_pwr": 0,
      "time_step": 1
    }
    """
    wind_speeds: List[float]   # list of horizontal wind speeds [m/s]
    rotor_speed: float         # rotor speed [rpm]
    gen_pwr: float             # generator power [kW]
    time_step: float = 1.0     # currently not used, but kept for future


class WindProfileResponse(BaseModel):
    """
    Response body for single-point wind-profile prediction.
    """
    pitch: float                        # predicted pitch angle [deg]
    model_name: str = "rf_wind_profile_model"


# ---- NEW: Time-series version (Option B) ----

class WindProfileSeriesRequest(BaseModel):
    """
    Request body for /predict_wind_profile_series (time series).

    All three lists MUST have the same length N:
    - wind_speeds[i]
    - rotor_speeds[i]
    - gen_powers[i]
    correspond to the same time instant.
    """
    wind_speeds: List[float]
    rotor_speeds: List[float]
    gen_powers: List[float]
    time_step: float = 1.0  # seconds between samples (for plotting)


class WindProfileSeriesResponse(BaseModel):
    """
    Response body for time-series wind-profile prediction.
    """
    pitch_series: List[float]           # list of pitch angles [deg]
    model_name: str = "rf_wind_profile_model"
