from pydantic import BaseModel
from typing import List

__all__ = ["TensorRequest", "TensorResponse"]

class TensorRequest(BaseModel):
    data: List[List[float]]

class TensorResponse(BaseModel):
    result: List[List[float]]
