import logging
from typing import Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn

from model.simple_gpt import SimpleGPT
from shared.dist_utils import setup_distributed
from shared.checkpoint import CheckpointManager
from dataset.text_dataset import TextDataset
from tokenizer import SentencePieceTokenizer


class Trainer:
    """Distributed training manager class."""

    def __init__(self, config: Dict[str, Any] = None, checkpoint_path: str = None):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ckpt_mgr = CheckpointManager(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = SentencePieceTokenizer(model_path=config['tokenizer']['model_path'])

        # Setup distributed training
        self.rank = setup_distributed()

        # Initialize model
        self.model = SimpleGPT.from_config(config['model'])
        self.model = self.model.to(self.device)

        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[self.device] if torch.cuda.is_available() else None)

        # Setup optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config['training']['learning_rate'])

        # Setup data loader
        self.train_loader = self._create_data_loader('train')
        self.val_loader = self._create_data_loader('val')

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # 체크포인트가 있으면 바로 로드
        if checkpoint_path:
            state = self.ckpt_mgr.load(self.model, self.optimizer, checkpoint_path)
            self.current_epoch = state['epoch']
            self.global_step = state['step']
            self.config = state['config']

    def _create_data_loader(self, split: str) -> DataLoader:
        """Create a data loader for the specified split."""
        # 데이터셋 설정
        dataset = TextDataset(
            jsonl_file=f'kakaotalk_{split}_data.jsonl',
            tokenizer=self.tokenizer,
            max_seq_len=self.config['model']['max_seq_len'],
            special_tokens=self.config['tokenizer']['special_tokens']
        )
        
        # DataLoader 설정
        loader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=(split == 'train'),  # train일 때만 shuffle
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True,  # GPU로 빠르게 전송하기 위해
            drop_last=True  # 마지막 불완전한 배치 제외
        )
        
        return loader

    def train(self, num_epochs: int):
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self._train_epoch()
            if self.rank == 0:
                self._validate_epoch()
                # 저장할 때 is_best 플래그 주거나 나중에 호출
                self.ckpt_mgr.save(self.model, self.optimizer, self.current_epoch, self.global_step)

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
        # outputs: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len]
        
        # CrossEntropyLoss를 위해 차원 조정
        # [batch_size * seq_len, vocab_size]
        outputs = outputs.view(-1, outputs.size(-1))
        # [batch_size * seq_len]
        labels = labels.view(-1)
        
        # 패딩 토큰의 loss는 무시
        criterion = nn.CrossEntropyLoss(
            ignore_index=-100  # TextDataset에서 패딩 토큰을 -100으로 설정
        )
        
        return criterion(outputs, labels)

    def save_checkpoint(self):
        """Save the current model state."""
        self.ckpt_mgr.save(
            self.model,
            self.optimizer,
            self.global_step,
            self.current_epoch,
            self.config
        )

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = self.ckpt_mgr.load(
            self.model,
            self.optimizer,
            checkpoint_path
        )

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
