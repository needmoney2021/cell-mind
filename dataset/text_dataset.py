from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    DEFAULT_SPECIAL_TOKENS: Dict[str, str] = {
        "url": "<|url|>",
        "photo": "<|photo|>",
        "video": "<|video|>",
    }

    def __init__(
        self,
        jsonl_file: str,
        tokenizer,
        max_seq_len: int,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()

        self.special_tokens = self.DEFAULT_SPECIAL_TOKENS.copy()
        if special_tokens:
            self.special_tokens.update(special_tokens)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.bos_id = tokenizer.piece_to_id(self.special_tokens["bos"])
        self.eos_id = tokenizer.piece_to_id(self.special_tokens["eos"])
        self.pad_id = (
            tokenizer.pad_id()
            if tokenizer.pad_id() is not None
            else (tokenizer.unk_id() if tokenizer.unk_id() is not None else 0)
        )

        self.seqs: List[List[int]] = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                text: str = raw["text"]

                ids = tokenizer.encode_as_ids(text)
                ids = [self.bos_id] + ids + [self.eos_id]

                ids = ids[: self.max_seq_len]
                self.seqs.append(ids)

        print(f"[TextDataset] loaded {len(self.seqs)} sequences")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        ids = self.seqs[idx]

        if len(ids) < self.max_seq_len:
            ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        target_ids[target_ids == self.pad_id] = -100

        return input_ids, target_ids
