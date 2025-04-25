import torch.nn as nn

__all__ = ['TransformerBlock']

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model*4,
            activation='gelu',
            batch_first=True,
        )

    def forward(self, x):
        return self.block(x)