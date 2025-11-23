from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PitchRequest,
    PitchResponse,
    PitchBatchRequest,
    PitchBatchResponse,
)
from api.model_loader import predict_pitch_with_raw, predict_pitch_batch_with_raw

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


# ⬇️ NEW: batch endpoint
@app.post("/predict_batch", response_model=PitchBatchResponse)
def predict_batch(batch: PitchBatchRequest):
    """
    Batch prediction endpoint to avoid many small /predict calls.
    """
    ws_list = [s.wind_speed for s in batch.samples]
    rs_list = [s.rotor_speed for s in batch.samples]
    p_list  = [s.power for s in batch.samples]

    results_tuples = predict_pitch_batch_with_raw(ws_list, rs_list, p_list)

    results = [
        PitchResponse(pitch=clipped, pitch_raw=raw)
        for clipped, raw in results_tuples
    ]

    return PitchBatchResponse(results=results)
