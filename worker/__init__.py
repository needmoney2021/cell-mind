"""
RUN Worker Server
    > uvicorn worker:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
import torch
from shared import TensorRequest, TensorResponse
from shared import load_transformer_block

#

app = FastAPI()

D_MODEL = 64
N_HEAD = 2

block = load_transformer_block(D_MODEL, N_HEAD)
block.eval()

@app.post("/forward")
def forward(req: TensorRequest) -> TensorResponse:
    with torch.no_grad():
        x = torch.tensor(req.data, dtype=torch.float32)
        y = block(x)
        return TensorResponse(result=y.tolist())