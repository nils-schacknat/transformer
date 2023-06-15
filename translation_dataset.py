import torch
from torch.utils.data import Dataset, DataLoader

import os
from util import load_config, load_tokenizers, collate_fn


class TranslationDataset(Dataset):

    def __init__(self, source_file, target_file, source_tokenizer, target_tokenizer, save_processed_text=False):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_vocab_size = source_tokenizer.vocab_size()
        self.target_vocab_size = target_tokenizer.vocab_size()

        self.transformer_params = dict(source_vocab_size=self.source_vocab_size,
                                       target_vocab_size=self.target_vocab_size,
                                       bos_idx=self.target_tokenizer.bos_id(),
                                       eos_idx=self.target_tokenizer.eos_id())

        # Check if processed tensors are already saved
        source_tensors_file = f'{source_file}.pt'
        target_tensors_file = f'{target_file}.pt'
        if os.path.exists(source_tensors_file) and os.path.exists(target_tensors_file):
            self.source_sentences = torch.load(source_tensors_file)
            self.target_sentences = torch.load(target_tensors_file)
        else:
            self.source_sentences = self.process_sentences(source_file, self.source_tokenizer)
            self.target_sentences = self.process_sentences(target_file, self.target_tokenizer)

            if save_processed_text:
                torch.save(self.source_sentences, source_tensors_file)
                torch.save(self.target_sentences, target_tensors_file)

    @staticmethod
    def process_sentences(file_path, tokenizer):
        processed_sentences = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for sentence in file.readlines()[:30]:
                if sentence == "\n":
                    continue
                indices = tokenizer.encode(sentence.strip('\n'), add_bos=True, add_eos=True)
                processed_sentences.append(indices)

        return processed_sentences

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, index):
        source_indices = self.source_sentences[index]
        target_indices = self.target_sentences[index]

        return source_indices, target_indices

    def get_sentence(self, index):
        source_indices, target_indices = self[index]

        source_sentence = self.source_tokenizer.decode(source_indices)
        target_sentence = self.target_tokenizer.decode(target_indices)

        return source_sentence, target_sentence

    def get_tokens(self, index):
        source_sentence, target_sentence = self.get_sentence(index)

        source_tokens = self.source_tokenizer.encode_as_pieces(source_sentence)
        target_tokens = self.target_tokenizer.encode_as_pieces(target_sentence)

        return source_tokens, target_tokens


def load_german_english_translation_dataset():
    # Load the config
    config = load_config("config.yaml")

    # Load the tokenizers
    source_tokenizer, target_tokenizer = load_tokenizers(**config["tokenizer"])

    # Load the dataset
    dataset = TranslationDataset(**config["dataset"], source_tokenizer=source_tokenizer,
                                 target_tokenizer=target_tokenizer)

    return dataset


if __name__ == "__main__":
    # Load the config
    config = load_config("config.yaml")

    # Load the tokenizers
    source_tokenizer, target_tokenizer = load_tokenizers(**config["tokenizer"])

    # Load the dataset
    dataset = TranslationDataset(**config["dataset"], source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer)

    # Example sentence pair
    print("German:", dataset.get_sentence(1)[1])
    print("English:", dataset.get_sentence(1)[0])

    # Example sentence pair as human-readable tokens
    print("German:", dataset.get_tokens(1)[1])
    print("English:", dataset.get_tokens(1)[0])

    # Create a dataloader for batching
    data_loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)

    for batch in data_loader:
        input_batch, target_batch = batch
        print(input_batch)
        print(target_batch)
        break

