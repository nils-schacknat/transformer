import yaml
import sentencepiece as spm
import torch
from torch.utils.data import random_split, DataLoader


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
