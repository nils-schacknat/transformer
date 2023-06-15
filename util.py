import yaml
import sentencepiece as spm
import torch
from torch.utils.data import random_split


def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_tokenizers(source_tokenizer_path, target_tokenizer_path):
    source_tokenizer = spm.SentencePieceProcessor(model_file=source_tokenizer_path)
    target_tokenizer = spm.SentencePieceProcessor(model_file=target_tokenizer_path)

    return source_tokenizer, target_tokenizer


def split_dataset(dataset, test_size):
    # Calculate the sizes of train and test sets
    dataset_size = len(dataset)
    test_size = int(test_size * dataset_size)
    train_size = dataset_size - test_size

    # Randomly split the dataset into train and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def collate_fn(batch):
    inputs, outputs = zip(*batch)

    # Pad inputs and outputs to the length of the longest sequence
    max_length_inputs = max(len(seq) for seq in inputs)
    max_length_outputs = max(len(seq) for seq in outputs)

    padded_inputs = [seq + [-1] * (max_length_inputs - len(seq)) for seq in inputs]
    padded_outputs = [seq + [-1] * (max_length_outputs - len(seq)) for seq in outputs]

    return torch.IntTensor(padded_inputs), torch.IntTensor(padded_outputs)

