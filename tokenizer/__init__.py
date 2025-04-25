import argparse
import logging
import os
from typing import List, Optional

from .sentencepiece_tokenizer import SentencePieceTokenizer, TokenizerConfig

__all__ = ['SentencePieceTokenizer', 'TokenizerConfig']


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def train_tokenizer(
        input_file: str,
        output_dir: str,
        vocab_size: int = 8000,
        model_prefix: str = "tokenizer",
        character_coverage: float = 1.0,
        model_type: str = "bpe",
        control_tokens: Optional[List[str]] = None,
        user_defined_tokens: Optional[List[str]] = None
) -> None:
    """
    Train a new SentencePiece tokenizer.
    
    Args:
        input_file: Path to the input text file
        output_dir: Directory to save the tokenizer model
        vocab_size: Size of the vocabulary
        model_prefix: Prefix for the output model files
        character_coverage: Character coverage ratio
        model_type: Type of model (unigram, bpe, word, char)
        control_tokens: List of control tokens to add
        user_defined_tokens: List of user-defined tokens to add
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up tokenizer configuration
    config = TokenizerConfig(
        vocab_size=vocab_size,
        model_prefix=os.path.join(output_dir, model_prefix),
        character_coverage=character_coverage,
        model_type=model_type,
        control_tokens=control_tokens or [],
        user_defined_tokens=user_defined_tokens or []
    )

    # Initialize and train tokenizer
    tokenizer = SentencePieceTokenizer(config)

    # Save the trained model
    model_path = os.path.join(output_dir, f"{model_prefix}.model")
    tokenizer.save(model_path)

    logging.info(f"Tokenizer trained and saved to {model_path}")
    logging.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")


def main():
    """Main function for training the tokenizer."""
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer")
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--output-dir", default="tokenizer", help="Directory to save the tokenizer model")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Size of the vocabulary")
    parser.add_argument("--model-prefix", default="tokenizer", help="Prefix for the output model files")
    parser.add_argument("--character-coverage", type=float, default=1.0, help="Character coverage ratio")
    parser.add_argument("--model-type", default="bpe", choices=["unigram", "bpe", "word", "char"], help="Type of model")
    parser.add_argument("--control-tokens", nargs="+", help="List of control tokens to add")
    parser.add_argument("--user-defined-tokens", nargs="+", help="List of user-defined tokens to add")

    args = parser.parse_args()

    setup_logging()

    train_tokenizer(
        input_file=args.input,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        control_tokens=args.control_tokens,
        user_defined_tokens=args.user_defined_tokens
    )


if __name__ == "__main__":
    main()
