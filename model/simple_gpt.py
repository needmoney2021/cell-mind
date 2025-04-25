from typing import Dict, Optional

import torch
import torch.nn as nn

__all__ = ['SimpleGPT']


class SimpleGPT(nn.Module):
    """A simple GPT-like model implementation."""

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 6,
            max_seq_length: int = 1024,
            dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        # Transformer layers
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Output layer
        self.output = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Logits for next token prediction [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = input_ids.shape

        # Create position IDs
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        # Combine embeddings
        x = token_embeddings + position_embeddings

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self._generate_causal_mask(seq_length, device=input_ids.device)

        # Transformer forward pass
        x = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=attention_mask,
            memory_mask=attention_mask
        )

        # Output layer
        logits = self.output(x)

        return logits

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate a causal mask for the transformer."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    @classmethod
    def from_config(cls, config: Dict) -> 'SimpleGPT':
        """Create a model instance from a configuration dictionary."""
        return cls(
            vocab_size=config['vocab_size'],
            d_model=config.get('d_model', 512),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            max_seq_length=config.get('max_seq_length', 1024),
            dropout=config.get('dropout', 0.1)
        )
