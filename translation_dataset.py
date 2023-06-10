import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer

import os


class TranslationDataset(Dataset):
    def __init__(self, source_file, target_file, source_tokenizer, target_tokenizer, context_size):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_vocab_size = len(self.source_tokenizer.get_vocab())
        self.target_vocab_size = len(self.target_tokenizer.get_vocab())
        self.context_size = context_size
        self.return_tokens = False

        self.transformer_params = dict(source_vocab_size=self.source_vocab_size,
                                       target_vocab_size=self.target_vocab_size,
                                       context_size=self.context_size, pad_idx=self.target_tokenizer.pad_token_id,
                                       bos_idx=self.target_tokenizer.bos_token_id,
                                       eos_idx=self.target_tokenizer.eos_token_id)

        # Check if processed tensors are already saved
        source_tensors_file = f'{source_file}_{self.context_size}.pt'
        target_tensors_file = f'{target_file}_{self.context_size}.pt'
        if os.path.exists(source_tensors_file) and os.path.exists(target_tensors_file):
            self.source_sentences = torch.load(source_tensors_file)
            self.target_sentences = torch.load(target_tensors_file)
        else:
            self.banned_idx_list = []
            self.source_sentences = self.process_sentences(source_file, self.source_tokenizer)
            self.target_sentences = self.process_sentences(target_file, self.target_tokenizer)

            if self.banned_idx_list:
                mask = torch.ones(self.source_sentences.size(0), dtype=torch.bool)
                mask[self.banned_idx_list] = False
                self.source_sentences = self.source_sentences[mask]
                self.target_sentences = self.target_sentences[mask]

            torch.save(self.source_sentences, source_tensors_file)
            torch.save(self.target_sentences, target_tensors_file)

    def process_sentences(self, file_path, tokenizer):
        processed_sentences = []
        banned_idx_list = []
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = file.readlines()

            for i, sentence in enumerate(sentences):
                tokens = tokenizer.tokenize(sentence)
                tokens.insert(0, tokenizer.bos_token)
                tokens.append(tokenizer.eos_token)

                if len(tokens) > self.context_size:
                    tokens = tokens[:self.context_size]
                    banned_idx_list.append(i)

                padding_length = self.context_size - len(tokens)
                tokens.extend([tokenizer.pad_token] * padding_length)
                indices = tokenizer.convert_tokens_to_ids(tokens)
                processed_sentences.append(indices)

        self.banned_idx_list.extend(banned_idx_list)
        return torch.tensor(processed_sentences, dtype=torch.int)

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, index):
        source_indices = self.source_sentences[index]
        target_indices = self.target_sentences[index]

        if self.return_tokens:
            source_tokens = self.source_tokenizer.convert_ids_to_tokens(source_indices)
            target_tokens = self.target_tokenizer.convert_ids_to_tokens(target_indices)
            return source_tokens, target_tokens
        else:
            return source_indices, target_indices


def get_english_german_translation_dataset():
    context_size = 64
    source_file = 'translation_task/europarl-v7.de-en.en'
    target_file = 'translation_task/europarl-v7.de-en.de'

    source_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    source_tokenizer.add_special_tokens(dict(bos_token='[BOS]', eos_token='[EOS]'))

    target_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    target_tokenizer.add_special_tokens(dict(bos_token='[BOS]', eos_token='[EOS]'))

    dataset = TranslationDataset(source_file=source_file, target_file=target_file, source_tokenizer=source_tokenizer,
                                 target_tokenizer=target_tokenizer, context_size=context_size)

    return dataset


def split_dataset(dataset, test_size):
    # Calculate the sizes of train and test sets
    dataset_size = len(dataset)
    test_size = int(test_size * dataset_size)
    train_size = dataset_size - test_size

    # Randomly split the dataset into train and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Load the dataset
    dataset = get_english_german_translation_dataset()

    # Create a data loader for batching and shuffling
    batch_size = 32
    shuffle = False
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Example sentence pair in human-readable tokens
    dataset.return_tokens = True
    print("English:", dataset[1][0])
    print("German:", dataset[1][1])

    dataset.return_tokens = False
    # Iterate over the dataset during training
    for batch in data_loader:
        source_batch, target_batch = batch
        print(source_batch, target_batch)
        break
