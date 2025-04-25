import torch.nn as nn

from model import TransformerBlock

__all__ = ['load_transformer_block']


def load_transformer_block(d_model: int, n_head: int) -> nn.Module:
    return TransformerBlock(d_model=d_model, n_head=n_head)
