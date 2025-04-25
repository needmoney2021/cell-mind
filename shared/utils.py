import json
import logging
import os
from typing import Optional, Dict, Any

import psutil
import torch


def setup_logging(log_dir: str = 'logs') -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
    """
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for the project.
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})

    # Create data directory
    data_dir = paths.get('data', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Create checkpoints directory
    checkpoint_dir = paths.get('checkpoints', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create logs directory
    log_dir = paths.get('logs', 'logs')
    os.makedirs(log_dir, exist_ok=True)


def get_device(config: Dict[str, Any]) -> str:
    """
    Get device to use for training/inference.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    device = config.get('training', {}).get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        return 'cpu'
    return device


def get_batch_size(config: Dict[str, Any], device: str) -> int:
    """
    Get batch size based on device.
    
    Args:
        config: Configuration dictionary
        device: Device string ('cuda' or 'cpu')
        
    Returns:
        Batch size
    """
    training_config = config.get('training', {})
    if device == 'cuda':
        return training_config.get('gpu_batch_size', 32)
    return training_config.get('cpu_batch_size', 8)


def get_gradient_accumulation_steps(config: Dict[str, Any], device: str) -> int:
    """
    Get gradient accumulation steps based on device.
    
    Args:
        config: Configuration dictionary
        device: Device string ('cuda' or 'cpu')
        
    Returns:
        Number of gradient accumulation steps
    """
    training_config = config.get('training', {})
    if device == 'cuda':
        return training_config.get('gpu_gradient_accumulation_steps', 1)
    return training_config.get('cpu_gradient_accumulation_steps', 4)


def check_gpu_availability() -> bool:
    """
    Check if GPU is available.
    
    Returns:
        True if GPU is available, False otherwise
    """
    return torch.cuda.is_available()


def get_free_memory() -> Dict[str, float]:
    """
    Get available memory information.
    
    Returns:
        Dictionary containing memory information in GB
    """
    memory = psutil.virtual_memory()

    return {
        'total': memory.total / (1024 ** 3),
        'available': memory.available / (1024 ** 3),
        'used': memory.used / (1024 ** 3),
        'free': memory.free / (1024 ** 3)
    }


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Path to latest checkpoint file, or None if no checkpoints exist
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_') and f.endswith('.pt')
    ]

    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def get_worker_ips(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get worker IP addresses from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping worker ranks to IP addresses
    """
    return config.get('distributed', {}).get('worker_ips', {})


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary containing system information
    """
    return {
        'cpu': {
            'count': psutil.cpu_count(),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        },
        'memory': get_free_memory(),
        'gpu': {
            'available': check_gpu_availability(),
            'count': torch.cuda.device_count() if check_gpu_availability() else 0
        }
    }
