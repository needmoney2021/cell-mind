import os
from dataclasses import dataclass, field
from typing import List, Optional

import sentencepiece as spm


@dataclass
class TokenizerConfig:
    input_file: str = "input.txt"
    vocab_size: int = 8000
    model_prefix: str = "tokenizer"
    character_coverage: float = 1.0
    model_type: str = "bpe"
    control_tokens: List[str] = field(default_factory=list)
    user_defined_tokens: List[str] = field(default_factory=list)


class SentencePieceTokenizer:
    def __init__(
            self,
            config: Optional[TokenizerConfig] = None,
            model_path: Optional[str] = None
    ):
        self.config = config or TokenizerConfig()
        self.model_path = model_path

        self.sp = spm.SentencePieceProcessor()
        if model_path and os.path.exists(model_path):
            self.sp.Load(model_path)
        else:
            self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize the tokenizer with control and user-defined tokens"""
        # Prepare input file for training

        input_file = self.config.input_file

        # Train the model
        flags = " ".join([
            f"--input={input_file}",
            f"--model_prefix={self.config.model_prefix}",
            f"--vocab_size={self.config.vocab_size}",
            f"--character_coverage={self.config.character_coverage}",
            f"--model_type={self.config.model_type}",
            "--pad_id=0",
            "--unk_id=1",
            "--bos_id=2",
            "--eos_id=3",
            "--pad_piece=[pad]",
            "--unk_piece=[unk]",
            "--bos_piece=[bos]",
            "--eos_piece=[eos]",
        ])

        # user_defined_tokens 만으로도 동작하도록
        symbols = (self.config.user_defined_tokens or [])
        if symbols:
            flags += " --user_defined_symbols=" + ",".join(symbols)

        spm.SentencePieceTrainer.Train(flags)

        # Load the trained model
        self.sp.Load(f"{self.config.model_prefix}.model")

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        return self.sp.DecodeIds(ids)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens"""
        return self.sp.EncodeAsPieces(text)

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.sp.GetPieceSize()

    def save(self, path: str):
        """Save tokenizer model"""
        self.sp.Save(path)

    @classmethod
    def from_pretrained(cls, model_path: str) -> 'SentencePieceTokenizer':
        """Load a pretrained tokenizer"""
        return cls(model_path=model_path)

    def add_tokens(self, tokens: List[str]):
        """Add new tokens to the tokenizer"""
        if not hasattr(self, 'config'):
            self.config = TokenizerConfig()

        if not self.config.user_defined_tokens:
            self.config.user_defined_tokens = []

        self.config.user_defined_tokens.extend(tokens)
        self._initialize_tokenizer()
