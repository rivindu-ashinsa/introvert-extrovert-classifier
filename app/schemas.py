from pydantic import BaseModel, Field
from typing import List

# The model expects these features (raw, before preprocessing)
FEATURE_NAMES: List[str] = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]


class InputData(BaseModel):
    """Pydantic model for incoming prediction requests.

    These fields match the raw dataset columns (before preprocessing).
    All fields are required — ensure you supply the same names and types.
    """
    Time_spent_Alone: float = Field(..., description="Hours spent alone (numeric)")
    Stage_fear: str = Field(..., description="Stage fear (e.g., 'Yes' or 'No')")
    Social_event_attendance: float = Field(..., description="Number of social events attended (numeric)")
    Going_outside: float = Field(..., description="Frequency of going outside (numeric)")
    Drained_after_socializing: str = Field(..., description="Drained after socializing (e.g., 'Yes' or 'No')")
    Friends_circle_size: float = Field(..., description="Size of friend circle (numeric)")
    Post_frequency: float = Field(..., description="Posting frequency (numeric)")

    class Config:
        schema_extra = {
            "example": {
                "Time_spent_Alone": 3.5,
                "Stage_fear": "No",
                "Social_event_attendance": 4.0,
                "Going_outside": 3.0,
                "Drained_after_socializing": "Yes",
                "Friends_circle_size": 5.0,
                "Post_frequency": 2.0,
            }
        }




class PredictionResponse(BaseModel):
    prediction: str
