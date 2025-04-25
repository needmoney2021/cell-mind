import torch
import requests
from shared import endpoints, TensorRequest
from model import SimpleGPT

D_MODEL = 64
N_HEAD = 2
NUM_LAYERS = 2

model = SimpleGPT(
    vocab_size=8000,
    d_model=D_MODEL,
    n_head=N_HEAD,
    num_layers=NUM_LAYERS,
    max_seq_len=128,
)

model.eval()

def send_to_worker(endpoint: str, tensor: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    tensor_data = tensor.detach().cpu().tolist()
    req = TensorRequest(data=tensor_data, mask=mask.detach().cpu().tolist())
    res = requests.post(f"{endpoint}/forward", json=req.model_dump())
    res.raise_for_status()
    result_data = res.json()["result"]
    return torch.tensor(result_data, dtype=torch.float32)


def distributed_forward(input_tokens: torch.Tensor) -> torch.Tensor:

    with torch.no_grad():
        embedded = model.token_embedding(input_tokens) * (model.d_model ** 0.5)

        x = embedded

        for idx, endpoint in enumerate(endpoints):
            print(f"Sending to worker {idx + 1}: {endpoint}")
            x = send_to_worker(endpoint, x)

        logits = model.fc_out(x)

        return logits


if __name__ == "__main__":
    prompt = torch.randint(0, 7999, (1, 10), dtype=torch.long)

    output = distributed_forward(prompt)

    print("Distributed output shape:", output.shape)