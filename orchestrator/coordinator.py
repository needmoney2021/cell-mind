import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.multiprocessing as mp

from model import SimpleGPT
from tokenizer import SentencePieceTokenizer
from worker.trainer import DistributedTrainer


class TrainingCoordinator:
    def __init__(
            self,
            model_config: Dict[str, Any],
            training_config: Dict[str, Any],
            tokenizer_config: Dict[str, Any],
            num_gpus: Optional[int] = None
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer_config = tokenizer_config

        # Load distributed configuration from config.json
        config_path = Path(__file__).parent.parent / 'shared' / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.dist_config = config['distributed']

        self.num_gpus = num_gpus or len(self.dist_config["worker_ips"])

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Log coordinator configuration
        self.logger.info(f"Coordinator initialized with:")
        self.logger.info(f"  Master Address: {self.dist_config['init_method'].split('://')[1].split(':')[0]}")
        self.logger.info(f"  Master Port: {self.dist_config['init_method'].split(':')[-1]}")
        self.logger.info(f"  World Size: {self.dist_config['world_size']}")
        self.logger.info(f"  Number of Workers: {len(self.dist_config['worker_ips'])}")

    def _setup_tokenizer(self) -> SentencePieceTokenizer:
        """Initialize and setup the tokenizer."""
        tokenizer = SentencePieceTokenizer(
            config=self.tokenizer_config
        )
        return tokenizer

    def _setup_model(self) -> SimpleGPT:
        """Initialize the model."""
        model = SimpleGPT(**self.model_config)
        return model

    def _worker_process(
            self,
            rank: int,
            world_size: int,
            model: SimpleGPT,
            train_dataset: torch.utils.data.Dataset,
            tokenizer: SentencePieceTokenizer
    ) -> None:
        """Worker process for distributed training."""
        # Get worker configuration
        worker_ip = self.dist_config["worker_ips"][rank]
        username = self.dist_config["usernames"][worker_ip]

        # Set environment variables for this process
        os.environ['MASTER_ADDR'] = self.dist_config['init_method'].split('://')[1].split(':')[0]
        os.environ['MASTER_PORT'] = self.dist_config['init_method'].split(':')[-1]
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup optimizer and criterion
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training_config.get('learning_rate', 1e-4)
        )
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize trainer with distributed configuration
        trainer = DistributedTrainer(
            model=model,
            train_dataset=train_dataset,
            optimizer=optimizer,
            criterion=criterion,
            config={
                **self.training_config,
                'master_addr': self.dist_config['init_method'].split('://')[1].split(':')[0],
                'master_port': self.dist_config['init_method'].split(':')[-1],
                'backend': self.dist_config['backend']
            },
            rank=rank,
            world_size=world_size
        )

        # Start training
        trainer.train(self.training_config.get('num_epochs', 10))

    def start_training(self, train_dataset: torch.utils.data.Dataset) -> None:
        """Start distributed training."""
        # Setup tokenizer
        tokenizer = self._setup_tokenizer()

        # Setup model
        model = self._setup_model()

        # Start distributed training
        world_size = len(self.dist_config['worker_ips'])
        if world_size > 1:
            self.logger.info(f"Starting distributed training on {world_size} workers")
            mp.spawn(
                self._worker_process,
                args=(world_size, model, train_dataset, tokenizer),
                nprocs=world_size,
                join=True
            )
        else:
            self.logger.info("Starting training on single worker")
            self._worker_process(0, 1, model, train_dataset, tokenizer)

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        # This would typically be called after training is complete
        # Implementation depends on how you want to handle model saving
        pass

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        # Implementation depends on how you want to handle model loading
        pass
