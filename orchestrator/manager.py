import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
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


def run_command_on_worker(worker_ip: str, command: str, username: str) -> bool:
    """
    Run a command on a worker node using SSH.
    
    Args:
        worker_ip: IP address of the worker node
        command: Command to run
        username: Username for SSH connection
        
    Returns:
        True if command was successful, False otherwise
    """
    try:
        subprocess.run([
            'ssh', f'{username}@{worker_ip}',
            command
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command on {worker_ip}: {e}")
        return False


def distribute_checkpoint(checkpoint_path: str, worker_ip: str, username: str) -> bool:
    """
    Distribute a checkpoint to a worker node.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        worker_ip: IP address of the worker node
        username: Username for SSH connection
        
    Returns:
        True if distribution was successful, False otherwise
    """
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return False

    try:
        subprocess.run([
            'scp', checkpoint_path,
            f'{username}@{worker_ip}:{checkpoint_path}'
        ], check=True)
        logging.info(f"Successfully distributed checkpoint to {worker_ip}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error distributing checkpoint to {worker_ip}: {e}")
        return False


def start_training():
    """Start distributed training on all workers."""
    setup_logging()
    config = get_config()

    # Set environment variables for each worker
    for rank, worker_ip in enumerate(config['distributed']['worker_ips'], start=1):
        username = config['distributed']['usernames'][worker_ip]
        env_vars = {
            'MASTER_ADDR': config['distributed']['init_method'].split('://')[1].split(':')[0],
            'MASTER_PORT': config['distributed']['init_method'].split(':')[-1],
            'WORLD_SIZE': str(config['distributed']['world_size']),
            'RANK': str(rank)
        }

        env_str = ' '.join(f'{k}={v}' for k, v in env_vars.items())
        command = f'{env_str} python -m worker.processor --mode train'

        run_command_on_worker(worker_ip, command, username)


def start_inference(prompt: str):
    """Start distributed inference on all workers."""
    setup_logging()
    config = get_config()

    # Save prompt to file
    prompt_file = 'data/prompt.txt'
    with open(prompt_file, 'w') as f:
        f.write(prompt)

    # Get latest checkpoint
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
    if not checkpoints:
        logging.error("No checkpoints found")
        return

    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)

    # Distribute prompt and checkpoint to workers
    for worker_ip in config['distributed']['worker_ips']:
        username = config['distributed']['usernames'][worker_ip]
        distribute_file(prompt_file, worker_ip, username)
        distribute_checkpoint(str(latest_checkpoint), worker_ip, username)

    # Start inference on workers
    for rank, worker_ip in enumerate(config['distributed']['worker_ips'], start=1):
        username = config['distributed']['usernames'][worker_ip]
        env_vars = {
            'MASTER_ADDR': config['distributed']['init_method'].split('://')[1].split(':')[0],
            'MASTER_PORT': config['distributed']['init_method'].split(':')[-1],
            'WORLD_SIZE': str(config['distributed']['world_size']),
            'RANK': str(rank)
        }

        env_str = ' '.join(f'{k}={v}' for k, v in env_vars.items())
        command = f'{env_str} python -m worker.processor --mode inference --prompt-file {prompt_file}'

        run_command_on_worker(worker_ip, command, username)


def sync_checkpoints():
    """Sync checkpoints from workers to master."""
    setup_logging()
    config = get_config()

    for worker_ip in config['distributed']['worker_ips']:
        username = config['distributed']['usernames'][worker_ip]
        checkpoint_dir = config['paths']['checkpoint_dir']

        try:
            subprocess.run([
                'scp', '-r',
                f'{username}@{worker_ip}:{checkpoint_dir}/*',
                f'{checkpoint_dir}/'
            ], check=True)
            logging.info(f"Successfully synced checkpoints from {worker_ip}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error syncing checkpoints from {worker_ip}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'inference', 'sync'], required=True)
    parser.add_argument('--prompt', help='Prompt for inference mode')
    args = parser.parse_args()

    if args.mode == 'train':
        start_training()
    elif args.mode == 'inference':
        if not args.prompt:
            parser.error("--prompt is required for inference mode")
        start_inference(args.prompt)
    elif args.mode == 'sync':
        sync_checkpoints()
