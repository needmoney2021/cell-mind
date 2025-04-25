import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist

from .utils import setup_logging


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_distributed() -> None:
    """
    Set up distributed training environment using config.json.
    """
    # Load configuration from config.json
    config = load_config()
    dist_config = config.get('distributed', {})
    backend = dist_config.get('backend', 'gloo')
    init_method = dist_config.get('init_method', 'env://')
    world_size = dist_config.get('world_size', 1)
    rank = dist_config.get('rank', 0)

    # Set environment variables
    os.environ['MASTER_ADDR'] = dist_config.get('master_addr', 'localhost')
    os.environ['MASTER_PORT'] = str(dist_config.get('master_port', 29500))
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )

    # Set up logging
    log_dir = config.get('paths', {}).get('logs', 'logs')
    setup_logging(log_dir, f'worker_{rank}')

    logging.info(f"Initialized distributed training with backend {backend}")
    logging.info(f"World size: {world_size}, Rank: {rank}")


def cleanup_distributed() -> None:
    """
    Clean up distributed training environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("Cleaned up distributed training environment")


def is_main_process() -> bool:
    """
    Check if current process is the main process.
    
    Returns:
        True if current process is the main process, False otherwise
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    """
    Get the number of processes in the distributed group.
    
    Returns:
        Number of processes
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get the rank of the current process.
    
    Returns:
        Rank of the current process
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor across all processes and compute mean.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(get_world_size())
    return tensor


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source process to all other processes.
    
    Args:
        tensor: Tensor to broadcast
        src: Source process rank
        
    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def gather_tensors(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        Gathered tensors or None if not main process
    """
    if not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    if world_size == 1:
        return tensor

    # Create output tensor
    output = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor)

    if is_main_process():
        return torch.cat(output, dim=0)
    return None
