from pydantic import BaseModel
from typing import List, Optional

__all__ = ["TensorRequest", "TensorResponse"]

class TensorRequest(BaseModel):
    data: List[List[float]]
    mask: Optional[List[List[float]]] = None

class TensorResponse(BaseModel):
    result: List[List[float]]
