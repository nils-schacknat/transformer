import copy
import fileinput

from torchdata.datapipes.iter import IterableWrapper, FileOpener
import torchtext.transforms as T
import torch

from util import load_config, load_tokenizers
from typing import List, Tuple


class TranslationDatapipe:
    def __init__(
        self,
        source_file: str,
        target_file: str,
        source_tokenizer_path: str,
        target_tokenizer_path: str,
        batch_size: int,
    ):
        """
        Initializes the TranslationDatapipe.

        Args:
            source_file (str): Path to the source text file.
            target_file (str): Path to the target text file.
            source_tokenizer_path (str): Path to the source tokenizer model.
            target_tokenizer_path (str): Path to the target tokenizer model.
            batch_size (int): Batch size for the datapipe.
        """
        self.batch_size = batch_size

        # Load the tokenizers
        self.source_tokenizer, self.target_tokenizer = load_tokenizers(
            source_tokenizer_path=source_tokenizer_path,
            target_tokenizer_path=target_tokenizer_path,
        )

        # Data pipeline for source and target text
        self.source_datapipe = self.create_datapipe(
            file=source_file, tokenizer=self.source_tokenizer
        )
        self.target_datapipe = self.create_datapipe(
            file=target_file, tokenizer=self.target_tokenizer
        )

        # Combined, filtered, bucketed and padded datapipe
        self.datapipe = self.combine_datapipes(
            source_datapipe=self.source_datapipe, target_datapipe=self.target_datapipe
        )
        self.datapipe = self.datapipe.set_length(self.compute_len(source_file=source_file, target_file=target_file))

        # Parameters of the tokenizers
        self.tokenizer_params = dict(
            source_vocab_size=self.source_tokenizer.vocab_size(),
            target_vocab_size=self.target_tokenizer.vocab_size(),
            bos_idx=self.target_tokenizer.bos_id(),
            eos_idx=self.target_tokenizer.eos_id(),
        )

    @staticmethod
    def compute_len(source_file: str, target_file: str):
        total_len = 0
        for line_src, line_tgt in zip(fileinput.input(source_file), fileinput.input(target_file)):
            if line_src.strip() != '' and line_tgt.strip() != '':
                total_len += 1
        return total_len

    @staticmethod
    def create_datapipe(file: str, tokenizer) -> IterableWrapper:
        """
        Creates a data pipeline for a file using the given tokenizer.

        Args:
            file (str): Path to the file.
            tokenizer: Tokenizer object.

        Returns:
            IterableWrapper: Data pipeline for the file.
        """
        datapipe = IterableWrapper([file])
        datapipe = FileOpener(datapipe, mode="rb")
        datapipe = datapipe.readlines(decode=True, return_path=False)
        datapipe = datapipe.map(
            lambda x: tokenizer.encode(x.lower(), add_bos=True, add_eos=True)
        )
        return datapipe

    def combine_datapipes(
        self, source_datapipe: IterableWrapper, target_datapipe: IterableWrapper
    ) -> IterableWrapper:
        """
        Combines and preprocesses source and target data pipelines.

        Args:
            source_datapipe (IterableWrapper): Data pipeline for the source text.
            target_datapipe (IterableWrapper): Data pipeline for the target text.

        Returns:
            IterableWrapper: Combined and preprocessed data pipeline.
        """
        # Combine source and target pipelines
        datapipe = source_datapipe.zip(target_datapipe)

        # Filter out empty sequences
        datapipe = datapipe.filter(filter_fn=lambda x: len(x[0]) > 2 and len(x[1]) > 2)

        # Bucketing and batching
        datapipe = datapipe.bucketbatch(
            batch_size=self.batch_size,
            batch_num=64,
            bucket_num=1,
            use_in_batch_shuffle=False,
            sort_key=lambda bucket: sorted(
                bucket, key=lambda x: (len(x[0]), len(x[1]))
            ),
        )

        # Unzip the sequence pairs into separate source and target sequences
        datapipe = datapipe.map(lambda sequence_pairs: tuple(zip(*sequence_pairs)))

        def apply_padding(
            pair_of_sequences: Tuple[List[int], List[int]]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Convert sequences to tensors and apply padding.

            Args:
                pair_of_sequences (Tuple[List[int], List[int]]): Pair of source and target sequences.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Padded source and target sequences as tensors.
            """
            return (
                T.ToTensor(self.source_tokenizer.pad_id())(list(pair_of_sequences[0])),
                T.ToTensor(self.target_tokenizer.pad_id())(list(pair_of_sequences[1])),
            )

        # Apply padding and convert sequences to tensors
        datapipe = datapipe.map(apply_padding)

        return datapipe

    def __iter__(self) -> "TranslationDatapipe":
        """
        Returns the iterator object for the datapipe.

        Returns:
            TranslationDatapipe: Iterator object.
        """
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the next batch from the datapipe.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of source and target sequences as tensors.
        """
        return next(iter(self.datapipe))

    def __len__(self):
        return len(self.datapipe)

    def random_split(self, p_test: float):
        assert 0 < p_test < 1
        train_pipe, test_pipe = self.datapipe.random_split(weights={"train": 1-p_test, "test": p_test}, seed=0)

        train_translation_datapipe = copy.deepcopy(self)
        train_translation_datapipe.datapipe = train_pipe

        test_translation_datapipe = copy.deepcopy(self)
        test_translation_datapipe.datapipe = test_pipe

        return train_translation_datapipe, test_translation_datapipe


if __name__ == "__main__":
    # Load the config
    config = load_config("config.yaml")

    # Usage:
    datapipe = TranslationDatapipe(**config["datapipe"], **config["tokenizer"])

    for sample in datapipe:
        source_batch, target_batch = sample
        print(*datapipe.source_tokenizer.decode(source_batch.tolist()), sep="\n")
        print()
        print(*datapipe.target_tokenizer.decode(target_batch.tolist()), sep="\n")
        print()
        break
