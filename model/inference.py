import logging

import torch
import torch.distributed as dist
from shared.types import ConfigType
from tokenizer.sentencepiece import SentencePieceTokenizer

from model.simple_gpt import SimpleGPT
from shared.dist_utils import setup_distributed
from shared.model_utils import load_checkpoint


class Inference:
    """Distributed inference manager class."""

    def __init__(self, config: ConfigType):
        """
        Initialize the inference manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup distributed inference
        self.rank = setup_distributed()

        # Initialize model and tokenizer
        self.model = SimpleGPT.from_config(config['model'])
        self.model = self.model.to(self.device)

        if dist.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device] if torch.cuda.is_available() else None
            )

        self.tokenizer = SentencePieceTokenizer(config['paths']['tokenizer_model'])

        # Load latest checkpoint
        self._load_latest_checkpoint()

    def _load_latest_checkpoint(self):
        """Load the latest model checkpoint."""
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        checkpoints = list(Path(checkpoint_dir).glob('checkpoint_*.pt'))

        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")

        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        checkpoint = load_checkpoint(str(latest_checkpoint))

        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint: {latest_checkpoint}")

    def generate(
            self,
            prompt: str,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Number of top-k tokens to consider
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            Generated text
        """
        self.model.eval()

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)

        # Generate tokens
        generated_ids = self._generate_tokens(
            input_ids,
            max_length,
            temperature,
            top_k,
            top_p
        )

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())

        return generated_text

    def _generate_tokens(
            self,
            input_ids: torch.Tensor,
            max_length: int,
            temperature: float,
            top_k: int,
            top_p: float
    ) -> torch.Tensor:
        """
        Generate tokens using the model.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Number of top-k tokens to consider
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            Generated token IDs
        """
        generated = input_ids

        for _ in range(max_length):
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(generated)
                next_token_logits = outputs[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat((generated, next_token), dim=1)

            # Stop if EOS token is generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return generated
