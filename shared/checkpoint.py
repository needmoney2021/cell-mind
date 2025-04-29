# shared/checkpoint.py
import os
import logging
from typing import Dict, Any, Optional
import torch


class CheckpointManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_dir = config.get('paths', {}).get('checkpoints', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_checkpoint_files(self):
        return sorted(
            (f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt')), key=lambda fn: os.path.getmtime(os.path.join(self.checkpoint_dir, fn))
        )

    def get_latest(self) -> Optional[str]:
        files = self._get_checkpoint_files()
        return os.path.join(self.checkpoint_dir, files[-1]) if files else None

    def save(self,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             epoch: int,
             step: int,
             is_best: bool = False
    ) -> str:
        ckpt = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch{epoch}_step{step}.pt')
        torch.save(ckpt, path)
        logging.info(f"Saved checkpoint to {path}")

        if is_best:
            best = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(ckpt, best)
            logging.info(f"Saved best-model to {best}")

        # cleanup old
        self.cleanup(keep_last_n=self.config.get('training', {}).get('keep_last_checkpoints', 3))
        return path

    def load(self,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        if checkpoint_path is None:
            checkpoint_path = self.get_latest()
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            logging.warning(f"No checkpoint found at {checkpoint_path}")
            return {'epoch': 0, 'step': 0, 'config': self.config}

        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        return {
            'epoch': ckpt.get('epoch', 0),
            'step':  ckpt.get('step',  0),
            'config': ckpt.get('config', self.config)
        }

    def sync(self, worker_ips: Dict[int, str], username: str):
        latest = self.get_latest()
        if not latest:
            logging.warning("No checkpoints to sync")
            return
        for rank, ip in worker_ips.items():
            remote = f"{username}@{ip}:{self.checkpoint_dir}"
            os.system(f"ssh {username}@{ip} 'mkdir -p {self.checkpoint_dir}'")
            os.system(f"scp {latest} {remote}/")
            logging.info(f"Synced checkpoint to worker {rank} @ {ip}")

    def cleanup(self, keep_last_n: int = 3):
        files = self._get_checkpoint_files()
        for old in files[:-keep_last_n]:
            try:
                os.remove(os.path.join(self.checkpoint_dir, old))
                logging.info(f"Removed old checkpoint: {old}")
            except OSError as e:
                logging.error(f"Failed to remove {old}: {e}")
