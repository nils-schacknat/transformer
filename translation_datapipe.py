import fileinput
import sentencepiece as spm
from math import sqrt

from torchdata.datapipes.iter import IterableWrapper, FileOpener
import torchtext.transforms as T
import torch

from typing import List, Tuple


def create_datapipe(
    file: str, tokenizer: spm.SentencePieceProcessor
) -> IterableWrapper:
    """
    Creates a datapipe for a textfile using the given tokenizer.

    Args:
        file (str): Path to the textfile.
        tokenizer (SentencePieceProcessor): Tokenizer for the text.

    Returns:
        IterableWrapper: Datapipe for the textfile.
    """
    datapipe = IterableWrapper([file])
    datapipe = FileOpener(datapipe, mode="rb")
    datapipe = datapipe.readlines(decode=True, return_path=False)
    datapipe = datapipe.map(
        lambda x: tokenizer.encode(x.lower(), add_bos=True, add_eos=True)
    )
    return datapipe


def apply_padding(
    pair_of_sequences: Tuple[List[int], List[int]], source_pad_id: int, target_pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert sequences to tensors and apply padding.

    Args:
        pair_of_sequences (Tuple[List[int], List[int]]): Pair of source and target sequences.
        source_pad_id (int): Index of the source padding token.
        target_pad_id (int): Index of the target padding token.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded source and target sequences as tensors.
    """
    return (
        T.ToTensor(source_pad_id)(list(pair_of_sequences[0])),
        T.ToTensor(target_pad_id)(list(pair_of_sequences[1])),
    )


def combine_datapipes_train(
    source_datapipe: IterableWrapper,
    target_datapipe: IterableWrapper,
    max_token_count: int,
    buffer_size: int,
    source_pad_id: int,
    target_pad_id: int,
) -> IterableWrapper:
    """
    Combines and preprocesses source and target data pipelines for training.

    Args:
        source_datapipe (IterableWrapper): Data pipeline for the source text.
        target_datapipe (IterableWrapper): Data pipeline for the target text.
        max_token_count (int): Maximum token count for each batch.
        buffer_size (int): Buffer size for batching.
        source_pad_id (int): Padding ID for source sequences.
        target_pad_id (int): Padding ID for target sequences.

    Returns:
        IterableWrapper: The processed datapipe for training.
    """
    # Combine source and target pipelines
    datapipe = source_datapipe.zip(target_datapipe)

    # Filter out empty sequences
    datapipe = datapipe.filter(filter_fn=lambda x: len(x[0]) > 2 and len(x[1]) > 2)

    # Bucketing and batching
    datapipe = datapipe.shuffle(buffer_size=buffer_size*3)
    datapipe = datapipe.max_token_bucketize(
        max_token_count=max_token_count,
        len_fn=lambda sample: sqrt(len(sample[0])**2 + len(sample[1])**2)*sqrt(2),
        buffer_size=buffer_size,
        include_padding=True
    )
    datapipe = datapipe.shuffle(buffer_size=256)

    # Unzip the sequence pairs into separate source and target sequences
    datapipe = datapipe.map(lambda sequence_pairs: tuple(zip(*sequence_pairs)))

    # Apply padding and convert sequences to tensors
    datapipe = datapipe.map(lambda x: apply_padding(x, source_pad_id, target_pad_id))

    return datapipe


def combine_datapipes_val(
    source_datapipe: IterableWrapper,
    target_datapipe: IterableWrapper,
    max_generation_length: int,
    max_token_count: int,
    buffer_size: int,
    source_pad_id: int,
    target_pad_id: int,
) -> IterableWrapper:
    """
    Combines and preprocesses source and target data pipelines for validation.

    Args:
        source_datapipe (IterableWrapper): Data pipeline for the source text.
        target_datapipe (IterableWrapper): Data pipeline for the target text.
        max_generation_length (int): The maximum length of generated sequences.
        max_token_count (int): Maximum token count for each batch.
        buffer_size (int): Buffer size for batching.
        source_pad_id (int): Padding ID for source sequences.
        target_pad_id (int): Padding ID for target sequences.

    Returns:
        IterableWrapper: The processed datapipe for validation.
    """
    # Combine source and target pipelines
    datapipe = source_datapipe.zip(target_datapipe)

    # Filter out empty sequences
    datapipe = datapipe.filter(filter_fn=lambda x: len(x[0]) > 2 and len(x[1]) > 2)

    # Bucketing and batching
    datapipe = datapipe.max_token_bucketize(
        max_token_count=max_token_count,
        max_len=max_generation_length,
        len_fn=lambda sample: len(sample[0]),
        buffer_size=buffer_size,
        include_padding=True
    )

    # Unzip the sequence pairs into separate source and target sequences
    datapipe = datapipe.map(lambda sequence_pairs: tuple(zip(*sequence_pairs)))

    # Apply padding and convert sequences to tensors
    datapipe = datapipe.map(lambda x: apply_padding(x, source_pad_id, target_pad_id))

    return datapipe


def create_translation_datapipe_train(
    source_tokenizer: spm.SentencePieceProcessor,
    target_tokenizer: spm.SentencePieceProcessor,
    source_file: str,
    target_file: str,
    max_token_count: int,
    buffer_size: int,
) -> IterableWrapper:
    """
    Creates a pipeline for training.

    Args:
        source_tokenizer (spm.SentencePieceProcessor): SentencePiece tokenizer for the source language.
        target_tokenizer (spm.SentencePieceProcessor): SentencePiece tokenizer for the target language.
        source_file (str): Path to the file containing the source text.
        target_file (str): Path to the file containing the target text.
        max_token_count (int): Maximum token count for each batch.
        buffer_size (int): Buffer size for batching.

    Returns:
        IterableWrapper: The processed datapipe for training.
    """
    source_datapipe = create_datapipe(file=source_file, tokenizer=source_tokenizer)
    target_datapipe = create_datapipe(file=target_file, tokenizer=target_tokenizer)

    # Combined, filtered, bucketed and padded datapipe
    datapipe = combine_datapipes_train(
        source_datapipe=source_datapipe,
        target_datapipe=target_datapipe,
        max_token_count=max_token_count,
        buffer_size=buffer_size,
        source_pad_id=source_tokenizer.pad_id(),
        target_pad_id=target_tokenizer.pad_id(),
    )

    return datapipe


def create_translation_datapipe_val(
    source_tokenizer: spm.SentencePieceProcessor,
    target_tokenizer: spm.SentencePieceProcessor,
    max_generation_length: int,
    source_file: str,
    target_file: str,
    max_token_count: int,
    buffer_size: int,
) -> IterableWrapper:
    """
    Creates a pipeline for validation.

    Args:
        source_tokenizer (spm.SentencePieceProcessor): SentencePiece tokenizer for the source language.
        target_tokenizer (spm.SentencePieceProcessor): SentencePiece tokenizer for the target language.
        max_generation_length (int): The maximum length of generated sequences.
        source_file (str): Path to the file containing the source text.
        target_file (str): Path to the file containing the target text.
        max_token_count (int): Maximum token count for each batch.
        buffer_size (int): Buffer size for batching.

    Returns:
        IterableWrapper: The processed datapipe for validation.
    """
    source_datapipe = create_datapipe(file=source_file, tokenizer=source_tokenizer)
    target_datapipe = create_datapipe(file=target_file, tokenizer=target_tokenizer)

    # Combined, filtered, bucketed, padded and split datapipes
    val_pipe = combine_datapipes_val(
        source_datapipe=source_datapipe,
        target_datapipe=target_datapipe,
        max_generation_length=max_generation_length,
        max_token_count=max_token_count,
        buffer_size=buffer_size,
        source_pad_id=source_tokenizer.pad_id(),
        target_pad_id=target_tokenizer.pad_id(),
    )

    return val_pipe


if __name__ == "__main__":
    from util import load_config, load_tokenizers
    import time

    # Load the config
    config = load_config("config.yaml")

    source_tokenizer, target_tokenizer = load_tokenizers(**config["tokenizer"])
    translation_datapipe = create_translation_datapipe_train(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        **config["datapipe_train"]
    )

    for source_batch, target_batch in translation_datapipe:
        num_src_tokens = torch.sum(source_batch != source_tokenizer.pad_id()).item()
        num_tgt_tokens = torch.sum(target_batch != target_tokenizer.pad_id()).item()
        print(f"src: batch_size={source_batch.shape[0]}, avg_token_percentage: {num_src_tokens/(source_batch.shape[0]*source_batch.shape[1]):.2f}")
        time.sleep(.5)
