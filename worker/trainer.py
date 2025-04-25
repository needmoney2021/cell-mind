import logging
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from shared.dist_utils import setup_distributed, cleanup_distributed, is_main_process, synchronize


def text_collate(batch):

    max_len = max(len(x[0]) for x in batch)

    pad_id = 0
    inputs, targets = [], []
    for inp, tgt in batch:
        # 뒤쪽 패딩
        inp = torch.nn.functional.pad(inp,   (0, max_len - len(inp)), value=pad_id)
        tgt = torch.nn.functional.pad(tgt,   (0, max_len - len(tgt)), value=-100)  # loss ignore
        inputs.append(inp)
        targets.append(tgt)

    return torch.stack(inputs), torch.stack(targets)


class DistributedTrainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataset: torch.utils.data.Dataset,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            config: Dict[str, Any],
            rank: int,
            world_size: int
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.rank = rank
        self.world_size = world_size

        # Setup distributed training
        setup_distributed(rank=rank, world_size=world_size)

        # Move model to GPU if available
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            output_device=rank if torch.cuda.is_available() else None
        )

        # Setup distributed sampler
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank
        )

        # Setup data loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.get('batch_size', 32),
            sampler=self.train_sampler,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=text_collate
        )

        # Setup logging
        if is_main_process():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        self.train_sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Reduce loss across all processes
            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= self.world_size

            total_loss += reduced_loss.item()

            if is_main_process() and batch_idx % self.config.get('log_interval', 10) == 0:
                self.logger.info(
                    f'Epoch: {epoch} [{batch_idx}/{num_batches}] '
                    f'Loss: {reduced_loss.item():.6f}'
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, num_epochs: int) -> None:
        """Train the model for specified number of epochs."""
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(epoch)

            if is_main_process():
                self.logger.info(f'Epoch {epoch} average loss: {avg_loss:.6f}')

            # Synchronize all processes
            synchronize()

        # Cleanup
        cleanup_distributed()

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        if is_main_process():
            checkpoint = {
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.config.get('current_epoch', 0),
                'config': self.config
            }
            torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        if is_main_process():
            checkpoint = torch.load(path)
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config['current_epoch'] = checkpoint['epoch']

        # Synchronize all processes after loading checkpoint
        synchronize()
