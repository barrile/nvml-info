from typing import Literal

from pydantic import BaseModel, Field

from .cuda import CUDA


class Device(BaseModel):
    type: Literal["CPU", "GPU"]
    required_memory: int = Field(default=8000, ge=0, strict=True)


__all__ = ["CUDA", "Device"]
