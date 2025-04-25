import requests
import torch
from .types import TensorRequest

__all__ = ["send_tensor"]

def send_tensor(endpoint: str, tensor: torch.Tensor) -> torch.Tensor:
    tensor_data = tensor.detach.cpu().tolist()
    req = TensorRequest(data=tensor_data)

    response = requests.post(endpoint, json=req.model_dump())
    response.raise_for_status()
    result = response.json()["result"]
    return torch.tensor(result, dtype=torch.float32)