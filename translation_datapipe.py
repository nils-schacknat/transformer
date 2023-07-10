import sentencepiece as spm
from math import sqrt

from datasets import load_dataset
from torchdata.datapipes.iter import IterableWrapper
import torchtext.transforms as T


def create_datapipe_test(
    test_dataset,
    tokenizer,
    max_token_count
) -> IterableWrapper:

    def get_tuple(dataset_entry):
        src = dataset_entry["translation"]["en"]
        tgt = dataset_entry["translation"]["de"]
        src_encoded = tokenizer.encode(src, add_bos=True, add_eos=True)

        return src_encoded, tgt

    # Create the datapipe from the dataset
    datapipe = IterableWrapper(test_dataset, deepcopy=False)
    datapipe = datapipe.map(get_tuple)

    # Bucketing and batching
    datapipe = datapipe.max_token_bucketize(
        max_token_count=max_token_count,
        len_fn=lambda sample: len(sample[0]),
        include_padding=True
    )

    # Unzip the sequence pairs into separate source and target sequences
    datapipe = datapipe.map(lambda sequence_pairs: tuple(zip(*sequence_pairs)))

    # Apply padding and convert sequences to tensors
    def apply_padding(pair_of_sequences):
        return (
            T.ToTensor(tokenizer.pad_id())(list(pair_of_sequences[0])),
            pair_of_sequences[1],
        )

    datapipe = datapipe.map(apply_padding)

    return datapipe


def create_datapipe_train(
    train_dataset,
    tokenizer: spm.SentencePieceProcessor,
    max_token_count: int,
) -> IterableWrapper:

    def get_processed_tuple(dataset_entry):
        src = dataset_entry["translation"]["en"]
        tgt = dataset_entry["translation"]["de"]

        src_encoded = tokenizer.encode(src, add_bos=True, add_eos=True)
        tgt_encoded = tokenizer.encode(tgt, add_bos=True, add_eos=True)

        keep_sequence_pair = tokenizer.unk_id() not in src_encoded and tokenizer.unk_id() not in tgt_encoded
        keep_sequence_pair &= .7 < len(src_encoded)/len(tgt_encoded) < 1/.7
        keep_sequence_pair &= len(src_encoded) > 5 and len(tgt_encoded) > 5
        keep_sequence_pair &= len(src_encoded) < 128 and len(tgt_encoded) < 128

        if keep_sequence_pair:
            return src_encoded, tgt_encoded

    # Create the datapipe from the dataset
    datapipe = IterableWrapper(train_dataset.shuffle(), deepcopy=True)
    datapipe = datapipe.map(get_processed_tuple)

    # Filter out invalid sequences
    datapipe = datapipe.filter(filter_fn=lambda x: x is not None)

    # Bucketing and batching
    datapipe = datapipe.max_token_bucketize(
        max_token_count=max_token_count,
        len_fn=lambda sample: sqrt(len(sample[0])**2 + len(sample[1])**2)/sqrt(2),
        include_padding=True,
        buffer_size=100000
    )
    datapipe = datapipe.shuffle(buffer_size=1000)

    # Unzip the sequence pairs into separate source and target sequences
    datapipe = datapipe.map(lambda sequence_pairs: tuple(zip(*sequence_pairs)))

    # Apply padding and convert sequences to tensors
    def apply_padding(pair_of_sequences):
        return (
            T.ToTensor(tokenizer.pad_id())(list(pair_of_sequences[0])),
            T.ToTensor(tokenizer.pad_id())(list(pair_of_sequences[1])),
        )
    datapipe = datapipe.map(apply_padding)

    return datapipe


def create_train_test_pipe(tokenizer: spm.SentencePieceProcessor, max_token_count: int):
    dataset = load_dataset("wmt14", "de-en")

    train_pipe = create_datapipe_train(
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        max_token_count=max_token_count
    )

    test_pipe = create_datapipe_test(
        test_dataset=dataset["test"],
        tokenizer=tokenizer,
        max_token_count=max_token_count
    )

    return train_pipe, test_pipe


if __name__ == "__main__":
    import torch
    from util import load_config, load_tokenizer

    # Load the config
    config = load_config("config.yaml")

    # Load the tokenizer
    tokenizer = load_tokenizer(**config["tokenizer"])

    # Load the datapipes
    train_pipe, test_pipe = create_train_test_pipe(
        tokenizer=tokenizer,
        max_token_count=config["max_token_count"]
    )

    for src_batch, tgt_batch in train_pipe:
        total_num_tokens = src_batch.shape[0] * src_batch.shape[1] + tgt_batch.shape[0] * tgt_batch.shape[1]
        num_padding_tokens = torch.sum(src_batch == tokenizer.pad_id()).item() + torch.sum(tgt_batch == tokenizer.pad_id()).item()
        print(src_batch.shape, tgt_batch.shape, round(num_padding_tokens/total_num_tokens, 2))
        print()
        print(*tokenizer.decode(src_batch.tolist())[:5], sep="\n")
        print()
        print(*tokenizer.decode(tgt_batch.tolist())[:5], sep="\n")
        print()
        break

    for src_batch, tgt_batch in test_pipe:
        print(*tokenizer.decode(src_batch.tolist())[:5], sep="\n")
        print()
        print(*tgt_batch[:5], sep="\n")
        print()
        break
