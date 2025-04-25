import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import GPUtil
import numpy as np
import psutil
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, loss: float, is_best: bool = False) -> str:
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint if applicable
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

        # Clean up old checkpoints
        self._cleanup()

        return str(checkpoint_path)

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint."""
        if path is None:
            # Load the best model if no path specified
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        return torch.load(path)

    def _cleanup(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > self.max_checkpoints:
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                checkpoint.unlink()


class MetricsTracker:
    """Tracks and logs training metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {}
        self.writer = None
        self.wandb_enabled = config['monitoring']['wandb']['enabled']

        # Initialize TensorBoard
        if config['monitoring']['tensorboard']:
            self.writer = SummaryWriter(config['paths']['tensorboard_dir'])

        # Initialize Weights & Biases
        if self.wandb_enabled:
            wandb.init(
                project=config['monitoring']['wandb']['project'],
                entity=config['monitoring']['wandb']['entity'],
                config=config
            )

    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics."""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(name, value, step)

            # Log to Weights & Biases
            if self.wandb_enabled:
                wandb.log({name: value}, step=step)

    def get_average(self, metric_name: str, window: int = None) -> float:
        """Get average of a metric over a window."""
        if metric_name not in self.metrics:
            return 0.0

        values = self.metrics[metric_name]
        if window is None:
            return np.mean(values)
        return np.mean(values[-window:])

    def close(self):
        """Close all logging resources."""
        if self.writer is not None:
            self.writer.close()
        if self.wandb_enabled:
            wandb.finish()


def setup_logging(config: Dict[str, Any], rank: int = 0) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}.log"

    logging.basicConfig(
        level=getattr(logging, config['monitoring']['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_gpu_memory_usage() -> Dict[int, float]:
    """Get GPU memory usage in MB."""
    gpus = GPUtil.getGPUs()
    return {gpu.id: gpu.memoryUsed for gpu in gpus}


def get_system_memory_usage() -> float:
    """Get system memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    optimizer_config = config['training']['optimizer']
    optimizer_type = optimizer_config['type'].lower()

    if optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            **optimizer_config['params']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler based on config."""
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type'].lower()

    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **scheduler_config['params']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def get_available_devices() -> List[Tuple[str, int]]:
    """Get list of available devices (both CPU and GPU).
    
    Returns:
        List of tuples (device_type, device_id)
        Example: [('cuda', 0), ('cuda', 1), ('cpu', 0)]
    """
    devices = []

    # Add GPUs if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices.extend([('cuda', i) for i in range(num_gpus)])

    # Always add CPU
    devices.append(('cpu', 0))

    return devices


def get_device(rank: int, config: Dict[str, Any]) -> Tuple[torch.device, str]:
    """Get appropriate device for the given rank.
    
    Args:
        rank: Process rank
        config: Configuration dictionary
        
    Returns:
        Tuple of (device, device_type)
    """
    devices = get_available_devices()

    # If no GPUs available, use CPU
    if not any(d[0] == 'cuda' for d in devices):
        return torch.device('cpu'), 'cpu'

    # Use GPU if available, otherwise fallback to CPU
    if rank < len(devices) - 1:  # -1 because we always have CPU
        device_type, device_id = devices[rank]
        return torch.device(f'{device_type}:{device_id}'), device_type
    else:
        return torch.device('cpu'), 'cpu'


def setup_mixed_precision(device_type: str) -> Tuple[torch.cuda.amp.GradScaler, bool]:
    """Setup mixed precision training if GPU is available."""
    use_amp = device_type == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    return scaler, use_amp


def get_memory_usage(device_type: str, device_id: int = 0) -> Dict[str, float]:
    """Get memory usage for the given device."""
    memory_info = {}

    if device_type == 'cuda' and torch.cuda.is_available():
        memory_info['gpu_memory'] = GPUtil.getGPUs()[device_id].memoryUsed
    else:
        memory_info['system_memory'] = psutil.Process().memory_info().rss / 1024 / 1024

    return memory_info
