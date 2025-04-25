import json
import logging
import os
import subprocess
from typing import Dict


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_config() -> Dict:
    """Load configuration from config.json."""
    with open('shared/config.json', 'r') as f:
        return json.load(f)


def distribute_file(file_path: str, worker_ip: str, username: str) -> bool:
    """
    Distribute a file to a worker node using scp.
    
    Args:
        file_path: Path to the file to distribute
        worker_ip: IP address of the worker node
        username: Username for SSH connection
        
    Returns:
        True if distribution was successful, False otherwise
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return False

    try:
        # Create necessary directories on worker
        remote_dir = os.path.dirname(file_path)
        subprocess.run([
            'ssh', f'{username}@{worker_ip}',
            f'mkdir -p {remote_dir}'
        ], check=True)

        # Copy file to worker
        subprocess.run([
            'scp', file_path,
            f'{username}@{worker_ip}:{file_path}'
        ], check=True)

        logging.info(f"Successfully distributed {file_path} to {worker_ip}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error distributing {file_path} to {worker_ip}: {e}")
        return False


def distribute_data():
    """Distribute data and tokenizer to all worker nodes."""
    setup_logging()
    config = get_config()

    # Files to distribute
    files_to_distribute = [
        'data/train.jsonl',
        'data/val.jsonl',
        'tokenizer.model',
        'tokenizer.vocab',
        'shared/config.json'
    ]

    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Distribute files to each worker
    for worker_ip in config['distributed']['worker_ips']:
        username = config['distributed']['usernames'][worker_ip]
        for file_path in files_to_distribute:
            distribute_file(file_path, worker_ip, username)


if __name__ == '__main__':
    distribute_data()
