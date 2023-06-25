import random

import yaml
import sentencepiece as spm
from typing import Dict, Tuple, List
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


def load_tokenizers(source_tokenizer_path: str, target_tokenizer_path: str) -> Tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
    """
    Load SentencePiece tokenizers.

    Args:
        source_tokenizer_path (str): Path to the source tokenizer model file.
        target_tokenizer_path (str): Path to the target tokenizer model file.

    Returns:
        Tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]: Loaded source and target tokenizers.
    """
    source_tokenizer = spm.SentencePieceProcessor(model_file=source_tokenizer_path)
    target_tokenizer = spm.SentencePieceProcessor(model_file=target_tokenizer_path)

    return source_tokenizer, target_tokenizer


def get_tokenizer_params(source_tokenizer: spm.SentencePieceProcessor, target_tokenizer: spm.SentencePieceProcessor) -> Dict[str, int]:
    """
    Get tokenizer parameters.

    Args:
        source_tokenizer (spm.SentencePieceProcessor): Source tokenizer.
        target_tokenizer (spm.SentencePieceProcessor): Target tokenizer.

    Returns:
        Dict[str, int]: Tokenizer parameters.
    """
    return dict(
        source_vocab_size=source_tokenizer.vocab_size(),
        target_vocab_size=target_tokenizer.vocab_size(),
        target_bos_id=target_tokenizer.bos_id(),
        target_eos_id=target_tokenizer.eos_id(),
        source_pad_id=source_tokenizer.pad_id()
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


def split_dataset_helper(input_file: str, train_file: str, val_file: str, p_val: float) -> None:
    """
    Splits the input file randomly into train and validation files based on the given percentage.

    Args:
        input_file (str): Path to the input file to be split.
        train_file (str): Path to the output train file.
        val_file (str): Path to the output validation file.
        p_val (float): The percentage of lines to be allocated for validation.

    Returns:
        None
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    random.seed(0)
    random.shuffle(lines)

    total_lines = len(lines)
    val_lines = int(total_lines * p_val)
    train_lines = total_lines - val_lines

    with open(train_file, 'w') as train:
        train.writelines(lines[:train_lines])

    with open(val_file, 'w') as val:
        val.writelines(lines[train_lines:])


if __name__ == "__main__":

    # Create a train - val split
    p_val = .05
    src_file = "translation_task/europarl-v7.de-en.de"
    tgt_file = "translation_task/europarl-v7.de-en.en"

    src_file_train = "translation_task/europarl-v7.de-en_train.de"
    tgt_file_train = "translation_task/europarl-v7.de-en_train.en"
    src_file_val = "translation_task/europarl-v7.de-en_val.de"
    tgt_file_val = "translation_task/europarl-v7.de-en_val.en"

    split_dataset_helper(src_file, src_file_train, src_file_val, p_val)
    split_dataset_helper(tgt_file, tgt_file_train, tgt_file_val, p_val)
