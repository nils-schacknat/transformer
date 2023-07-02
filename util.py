import yaml
import sentencepiece as spm
from typing import Dict, List
import torch


def load_config(path_to_config: str) -> Dict:
    """
    Load a YAML configuration file.

    Args:
        path_to_config (str): Path to the configuration file.

    Returns:
        Dict: Loaded configuration data.
    """
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_tokenizer(tokenizer_path: str) -> spm.SentencePieceProcessor:
    """
    Load SentencePiece tokenizers.

    Args:
        tokenizer_path (str): Path to the tokenizer model file.

    Returns:
        spm.SentencePieceProcessor: Loaded source and target tokenizers.
    """
    return spm.SentencePieceProcessor(model_file=tokenizer_path)


def get_tokenizer_params(tokenizer: spm.SentencePieceProcessor) -> Dict[str, int]:
    """
    Get tokenizer parameters.

    Args:
        tokenizer (spm.SentencePieceProcessor): The tokenizer.

    Returns:
        Dict[str, int]: Tokenizer parameters.
    """
    return dict(
        vocab_size=tokenizer.vocab_size(),
        bos_id=tokenizer.bos_id(),
        eos_id=tokenizer.eos_id(),
        pad_id=tokenizer.pad_id()
    )


def strip_tokens_after_eos(tensor: torch.Tensor, eos_id: int) -> torch.Tensor:
    """
    Strip tokens after the end-of-sequence (EOS) token in a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        eos_id (int): End-of-sequence (EOS) token ID.

    Returns:
        torch.Tensor: Tensor with tokens stripped after EOS.
    """
    tensor = tensor.tolist()
    for i, sequence in enumerate(tensor):
        if eos_id in sequence:
            tensor[i] = sequence[:sequence.index(eos_id)+1]
    return tensor


def translate_tensor(tensor: torch.Tensor, tokenizer: spm.SentencePieceProcessor) -> List[str]:
    """
    Translates a tensor using a tokenizer and returns the decoded strings.

    Args:
        tensor (torch.Tensor): The input tensor to be translated.
        tokenizer (spm.SentencePieceProcessor): The tokenizer used for decoding.

    Returns:
        List[str]: The decoded strings.
    """
    return tokenizer.decode(tensor.tolist())
