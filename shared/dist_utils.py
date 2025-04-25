import logging
import os
import socket
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist


def get_local_ip() -> str:
    """Get local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def setup_distributed(
        backend: str = 'nccl',
        init_method: str = 'env://',
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        timeout: int = 1800,  # 30 minutes
        comm_config: Optional[Dict[str, Any]] = None
) -> None:
    """Initialize the distributed environment with enhanced communication settings."""
    # Get environment variables if not provided
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

    if world_size is None or rank is None:
        raise ValueError("world_size and rank must be specified or set via environment variables")

    # Set default master address and port if not provided
    if master_addr is None:
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    if master_port is None:
        master_port = int(os.environ.get('MASTER_PORT', '29500'))

    # Configure communication settings
    if comm_config is None:
        comm_config = {}

    # Set environment variables for PyTorch distributed
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    # Additional communication settings
    if backend == 'nccl':
        # Enable NCCL debug if specified
        if comm_config.get('nccl_debug', False):
            os.environ['NCCL_DEBUG'] = 'INFO'

        # Set NCCL socket interface if specified
        if 'nccl_socket_ifname' in comm_config:
            os.environ['NCCL_SOCKET_IFNAME'] = comm_config['nccl_socket_ifname']

        # Set NCCL block time if specified
        if 'nccl_blocking_wait' in comm_config:
            os.environ['NCCL_BLOCKING_WAIT'] = str(comm_config['nccl_blocking_wait'])

    # Initialize process group with timeout
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=torch.timedelta(seconds=timeout)
    )

    # Log communication settings
    if rank == 0:  # Only log from master process
        logging.info(f"Distributed training initialized with:")
        logging.info(f"  Backend: {backend}")
        logging.info(f"  Master Address: {master_addr}")
        logging.info(f"  Master Port: {master_port}")
        logging.info(f"  World Size: {world_size}")
        logging.info(f"  Rank: {rank}")
        if comm_config:
            logging.info(f"  Communication Config: {comm_config}")


def cleanup_distributed() -> None:
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def get_world_size() -> int:
    """Get the number of processes in the distributed group."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    return get_rank() == 0


def synchronize() -> None:
    """Synchronize all processes."""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes."""
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt
