import logging
import os
from typing import Dict, Any, Optional

import torch

from .utils import get_latest_checkpoint


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        config: Dict[str, Any],
        is_best: bool = False
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        step: Current step
        config: Configuration dictionary
        is_best: Whether this is the best model so far
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = config.get('paths', {}).get('checkpoints', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }

    # Save checkpoint
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'checkpoint_epoch{epoch}_step{step}.pt'
    )
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)

    logging.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load checkpoint into
        optimizer: Optimizer to load checkpoint into
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        
    Returns:
        Dictionary containing epoch, step, and other checkpoint information
    """
    if checkpoint_path is None and config is not None:
        checkpoint_dir = config.get('paths', {}).get('checkpoints', 'checkpoints')
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        logging.warning(f"No checkpoint found at {checkpoint_path}")
        return {'epoch': 0, 'step': 0}

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return {
        'epoch': checkpoint['epoch'],
        'step': checkpoint['step'],
        'config': checkpoint.get('config', config)
    }


def sync_checkpoints(
        config: Dict[str, Any],
        worker_ips: Dict[str, str],
        username: str
) -> None:
    """
    Synchronize checkpoints between workers.
    
    Args:
        config: Configuration dictionary
        worker_ips: Dictionary mapping worker ranks to IP addresses
        username: SSH username for worker nodes
    """
    checkpoint_dir = config.get('paths', {}).get('checkpoints', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get latest checkpoint
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is None:
        logging.warning("No checkpoints found to sync")
        return

    # Sync checkpoints with each worker
    for rank, ip in worker_ips.items():
        try:
            # Create remote checkpoint directory
            remote_dir = f"{username}@{ip}:{checkpoint_dir}"
            os.system(f"ssh {username}@{ip} 'mkdir -p {checkpoint_dir}'")

            # Copy checkpoint
            os.system(f"scp {latest_checkpoint} {remote_dir}/")
            logging.info(f"Synced checkpoint to worker {rank} at {ip}")
        except Exception as e:
            logging.error(f"Failed to sync checkpoint with worker {rank}: {e}")


def cleanup_old_checkpoints(
        config: Dict[str, Any],
        keep_last_n: int = 3
) -> None:
    """
    Clean up old checkpoints, keeping only the most recent ones.
    
    Args:
        config: Configuration dictionary
        keep_last_n: Number of most recent checkpoints to keep
    """
    checkpoint_dir = config.get('paths', {}).get('checkpoints', 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return

    # Get all checkpoint files
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_') and f.endswith('.pt')
    ]

    # Sort by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))

    # Remove old checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        try:
            os.remove(os.path.join(checkpoint_dir, checkpoint))
            logging.info(f"Removed old checkpoint: {checkpoint}")
        except Exception as e:
            logging.error(f"Failed to remove checkpoint {checkpoint}: {e}")
