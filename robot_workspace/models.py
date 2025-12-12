# robot_workspace/models.py
from pydantic import BaseModel, Field, validator


class ObjectModel(BaseModel):
    label: str = Field(..., min_length=1)
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    width_m: float = Field(..., gt=0)
    height_m: float = Field(..., gt=0)

    @validator("width_m", "height_m")
    def check_reasonable_size(cls, v):
        if v > 1.0:  # Larger than 1 meter
            raise ValueError("Object size seems unreasonable")
        return v
