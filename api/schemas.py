from pydantic import BaseModel

class PitchRequest(BaseModel):
    wind_speed: float
    rotor_speed: float
    power: float

class PitchResponse(BaseModel):
    pitch: float
