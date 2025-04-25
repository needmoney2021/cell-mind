from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset

from shared.dist_utils import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    synchronize,
)
from model.simple_gpt import SimpleGPT
from worker.trainer import DistributedTrainer
from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(rank)s/%(world_size)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger("worker.processor")


class RandomLanguageDataset(Dataset):
    """꼭 교체하세요!

    랜덤 토큰으로 이뤄진 더미 데이터셋. 실제 프로젝트에서는 토크나이즈된
    `.pt` / `.npy` / `.bin` 등을 로딩하도록 바꿔야 합니다.
    """

    def __init__(self, vocab_size: int, seq_len: int = 128, length: int = 1024):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

    def __len__(self) -> int:  # type: ignore[override]
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_len,))
        return ids, ids.clone()  # (input_ids, labels)


def build_dataset(cfg: Dict[str, Any]) -> Dataset:
    """데이터셋 빌더.

    cfg 예시 (shared/config.json 안 혹은 외부 파일):
    ```json
    {
        "dataset": {
            "path": "/mnt/data/train.bin",
            "seq_len": 256,
            "length": 100000
        }
    }
    ```
    현재는 더미 랜덤셋을 반환한다.
    """
    seq_len = cfg.get("seq_len", 128)
    length = cfg.get("length", 1024)
    vocab_size = cfg.get("vocab_size", 8000)
    return RandomLanguageDataset(vocab_size=vocab_size, seq_len=seq_len, length=length)


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def run_train(config: Dict[str, Any]) -> None:
    """실제 학습 루프를 돌린다."""
    rank = get_rank()
    world_size = get_world_size()

    training_cfg = config.get("training", {})
    model_cfg: Dict[str, Any] = config.get("model", {})
    tokenizer_cfg: Dict[str, Any] = config.get("tokenizer", {})

    # ---------------------------------------------------------------------
    # Build tokenizer & model
    # ---------------------------------------------------------------------
    vocab_size = tokenizer_cfg.get("vocab_size", 8000)
    model = SimpleGPT(
        vocab_size=vocab_size,
        d_model=model_cfg.get("d_model", 512),
        nhead=model_cfg.get("nhead", 8),
        num_layers=model_cfg.get("num_layers", 6),
        dropout=model_cfg.get("dropout", 0.1),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # DDP wrapper (spawned per‑process -> device_ids 필요 여부 판단)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device] if torch.cuda.is_available() else None,
        )

    # ---------------------------------------------------------------------
    # Optimizer / Criterion
    # ---------------------------------------------------------------------
    lr = training_cfg.get("lr", 3e-4)
    betas = training_cfg.get("betas", (0.9, 0.95))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ---------------------------------------------------------------------
    # Dataset / Dataloader (sampler는 DistributedTrainer 에서 처리)
    # ---------------------------------------------------------------------
    dataset_cfg = training_cfg.get("dataset", {})
    train_dataset = build_dataset(dataset_cfg)

    trainer = DistributedTrainer(
        model=model,
        train_dataset=train_dataset,
        optimizer=optimizer,
        criterion=criterion,
        config=training_cfg,
        rank=rank,
        world_size=world_size,
    )

    epochs = training_cfg.get("epochs", 1)
    if is_main_process():
        logger.info(f"[Worker {rank}] starting training for {epochs} epoch(s)…")

    trainer.train(epochs)


@torch.no_grad()
def run_inference(prompt: str, config: Dict[str, Any]) -> None:
    """단일 프롬프트 분산 추론."""
    rank = get_rank()

    model_cfg: Dict[str, Any] = config.get("model", {})
    tokenizer_cfg: Dict[str, Any] = config.get("tokenizer", {})
    checkpoint_path: Optional[str] = model_cfg.get("checkpoint")

    # Build tokenizer
    tokenizer = SentencePieceTokenizer(**tokenizer_cfg) if tokenizer_cfg else SentencePieceTokenizer()
    vocab_size = tokenizer.vocab_size

    # Build & load model
    model = SimpleGPT(
        vocab_size=vocab_size,
        d_model=model_cfg.get("d_model", 512),
        nhead=model_cfg.get("nhead", 8),
        num_layers=model_cfg.get("num_layers", 6),
        dropout=model_cfg.get("dropout", 0.1),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        map_location = {"cuda:0": f"cuda:{rank}"} if torch.cuda.is_available() else "cpu"
        state = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) else state, strict=False)
        if is_main_process():
            logger.info(f"Loaded checkpoint → {checkpoint_path}")
    else:
        if is_main_process():
            logger.warning("Checkpoint not found. Using random‑init weights – 결과 품질이 낮을 수 있음.")

    # DDP wrapper (evaluation 모드)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device] if torch.cuda.is_available() else None,
        )
    model.eval()

    # Encode prompt & run forward
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits = model(input_ids)  # (1, seq_len, vocab)
    next_token = logits[:, -1].argmax(-1)  # greedy‑sample 한 토큰
    generated = torch.cat([input_ids.squeeze(0), next_token.cpu()])

    if is_main_process():
        decoded = tokenizer.decode(generated.tolist())
        print("================ Generated ================")
        print(decoded)
        print("===========================================")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:  # noqa: D401
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="CellMind Worker Processor")
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        required=True,
        help="Operation mode for this worker process.",
    )
    parser.add_argument("--prompt", help="Prompt string for inference mode.")
    parser.add_argument("--prompt-file", help="Path to prompt‑text file.")
    parser.add_argument("--config", default="shared/config.json", help="Path to JSON config.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:  # noqa: C901 – length OK here
    args = parse_args(argv)

    # ---------------------------------------------------------------------
    # Distributed 초기화 (MASTER_ADDR/PORT & RANK/WORLD_SIZE 는 환경변수로 전달됨)
    # ---------------------------------------------------------------------
    rank = setup_distributed()
    world_size = get_world_size()
    logging.LoggerAdapter(logger, {"rank": rank, "world_size": world_size})

    # ---------------------------------------------------------------------
    # config 로드
    # ---------------------------------------------------------------------
    if not Path(args.config).exists():
        logger.error(f"Config file {args.config} not found. 종료합니다.")
        sys.exit(1)

    with open(args.config, "r", encoding="utf‑8") as f:
        config: Dict[str, Any] = json.load(f)

    if args.mode == "train":
        run_train(config)

    elif args.mode == "inference":
        prompt: Optional[str] = None
        if args.prompt:
            prompt = args.prompt
        elif args.prompt_file:
            p_path = Path(args.prompt_file)
            if not p_path.exists():
                logger.error(f"Prompt file {p_path} not found")
                sys.exit(1)
            prompt = p_path.read_text(encoding="utf‑8")
        else:
            logger.error("--prompt 또는 --prompt-file 중 하나는 반드시 제공되어야 합니다.")
            sys.exit(1)

        run_inference(prompt, config)

    else:  # pragma: no cover – argparse 가 이미 검증하지만 살려둠
        logger.error(f"Unknown mode {args.mode}")
        sys.exit(1)

    synchronize()
    cleanup_distributed()


if __name__ == "__main__":
    main()
