import logging

import torch
import torch.distributed as dist
from shared.types import ConfigType
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.simple_gpt import SimpleGPT
from shared.dist_utils import setup_distributed
from shared.model_utils import load_checkpoint, save_checkpoint


class Trainer:
    """Distributed training manager class."""

    def __init__(self, config: ConfigType):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup distributed training
        self.rank = setup_distributed()

        # Initialize model
        self.model = SimpleGPT.from_config(config['model'])
        self.model = self.model.to(self.device)

        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[self.device] if torch.cuda.is_available() else None)

        # Setup optimizer
        self.optimizer = Adam(self.model.parameters(), lr=config['training']['learning_rate'])

        # Setup data loader
        self.train_loader = self._create_data_loader('train')
        self.val_loader = self._create_data_loader('val')

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def _create_data_loader(self, split: str) -> DataLoader:
        """Create a data loader for the specified split."""
        # TODO: Implement data loading logic
        pass

    def train(self, num_epochs: int):
        """
        Run the training loop.
        
        Args:
            num_epochs: Number of epochs to train for
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self._train_epoch()

            if self.rank == 0:  # Only validate on rank 0
                self._validate_epoch()

            if self.rank == 0:  # Only save checkpoints on rank 0
                self.save_checkpoint()

    def _train_epoch(self):
        """Run one training epoch."""
        self.model.train()

        for batch in self.train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(input_ids)
            loss = self._compute_loss(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            if self.global_step % self.config['training']['logging_steps'] == 0:
                logging.info(f"Step {self.global_step}, Loss: {loss.item():.4f}")

    def _validate_epoch(self):
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids)
                loss = self._compute_loss(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        logging.info(f"Validation Loss: {avg_loss:.4f}")

    def _compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss between model outputs and labels."""
        # TODO: Implement loss computation
        pass

    def save_checkpoint(self):
        """Save the current model state."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        save_checkpoint(
            checkpoint,
            self.config['paths']['checkpoint_dir'],
            f"checkpoint_epoch{self.current_epoch}_step{self.global_step}.pt"
        )

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = load_checkpoint(checkpoint_path)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
