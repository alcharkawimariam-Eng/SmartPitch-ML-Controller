from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PitchRequest,
    PitchResponse,
    PitchBatchRequest,
    PitchBatchResponse,
    WindProfileRequest,
    WindProfileResponse,
    WindProfileSeriesRequest,       # NEW
    WindProfileSeriesResponse,      # NEW
)
from api.model_loader import predict_pitch_with_raw, predict_pitch_batch_with_raw
from api.model_loader_wind_profile import (
    predict_wind_profile_pitch,
    predict_wind_profile_series,    # NEW
)


app = FastAPI(
    title="SmartPitch API",
    description="AI-powered pitch angle prediction for Region 3 wind turbines.",
    version="1.0.0"
)

# CORS (optional, but good for UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.get("/model_info")
def model_info():
    return {
        "model": "MLPRegressor",
        "features": ["wind_speed", "rotor_speed", "power"],
        "target": "pitch",
        "version": "1.0"
    }


@app.post("/predict", response_model=PitchResponse)
def predict(req: PitchRequest):
    clipped, raw = predict_pitch_with_raw(
        req.wind_speed,
        req.rotor_speed,
        req.power,
    )
    return PitchResponse(
        pitch=clipped,
        pitch_raw=raw,
    )


# ⬇️ Batch endpoint
@app.post("/predict_batch", response_model=PitchBatchResponse)
def predict_batch(batch: PitchBatchRequest):
    """
    Batch prediction endpoint to avoid many small /predict calls.
    """
    ws_list = [s.wind_speed for s in batch.samples]
    rs_list = [s.rotor_speed for s in batch.samples]
    p_list = [s.power for s in batch.samples]

    results_tuples = predict_pitch_batch_with_raw(ws_list, rs_list, p_list)

    results = [
        PitchResponse(pitch=clipped, pitch_raw=raw)
        for clipped, raw in results_tuples
    ]

    return PitchBatchResponse(results=results)


# ⬇️ Single-point wind-profile endpoint (KEEP)
@app.post("/predict_wind_profile", response_model=WindProfileResponse)
def predict_wind_profile(req: WindProfileRequest):
    """
    Predict Region 3 blade pitch using the wind-profile Random Forest model
    for a SINGLE operating point.

    Request (from schema):
    - wind_speeds: List[float]  -> we use the first element for now
    - rotor_speed: float
    - gen_pwr: float
    - time_step: float
    """
    # Make sure we have at least one wind speed
    if not req.wind_speeds:
        raise HTTPException(status_code=400, detail="wind_speeds list must not be empty")

    # Take the first wind speed from the list
    hor_windv = req.wind_speeds[0]

    pitch = predict_wind_profile_pitch(
        hor_windv,
        req.rotor_speed,
        req.gen_pwr,
    )
    return WindProfileResponse(pitch=pitch)


# ⬇️ NEW: Time-series wind-profile endpoint
@app.post("/predict_wind_profile_series", response_model=WindProfileSeriesResponse)
def predict_wind_profile_series_endpoint(req: WindProfileSeriesRequest):
    """
    Predict Region 3 blade pitch using the wind-profile model for a FULL TIME SERIES.

    All lists must have the same length N.
    """
    n_ws = len(req.wind_speeds)
    n_rs = len(req.rotor_speeds)
    n_gp = len(req.gen_powers)

    if n_ws == 0:
        raise HTTPException(status_code=400, detail="wind_speeds list must not be empty")
    if not (n_ws == n_rs == n_gp):
        raise HTTPException(
            status_code=400,
            detail="wind_speeds, rotor_speeds and gen_powers must have the SAME length",
        )

    pitch_list = predict_wind_profile_series(
        req.wind_speeds,
        req.rotor_speeds,
        req.gen_powers,
    )

    return WindProfileSeriesResponse(pitch_series=pitch_list)
