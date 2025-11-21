from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PitchRequest, PitchResponse
from api.model_loader import predict_pitch

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
    pitch = predict_pitch(req.wind_speed, req.rotor_speed, req.power)
    return PitchResponse(pitch=pitch)
