import torch
import torch.nn as nn
from transformer_block import TransformerBlock

__all__ = ['SimpleGPT']

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.decode_layer = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def generate_causal_mask(self, size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tokens):
        batch_size, seq_len = tokens.size()
        causal_mask = self.generate_causal_mask(seq_len).to(tokens.device)

        x = self.token_embedding(tokens) * (self.d_model ** 0.5)

        for layer in self.decode_layer:
            x = layer(x, mask=causal_mask)

        logits = self.fc_out(x)
        return logits