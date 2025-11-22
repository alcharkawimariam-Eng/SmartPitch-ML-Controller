from pydantic import BaseModel 
from typing import Optional

class PitchRequest(BaseModel):
    wind_speed: float
    rotor_speed: float
    power: float

class PitchResponse(BaseModel):
    pitch: float                # clipped (safe) command
    pitch_raw: Optional[float] = None  # ⬅ ADD THIS DEFAULT
