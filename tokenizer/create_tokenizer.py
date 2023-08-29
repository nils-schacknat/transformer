"""
A script for creating a SentencePiece tokenizer from the dataset for both the source and the target language.
"""

from datasets import load_dataset
import sentencepiece as spm

train_data = load_dataset("wmt14", "de-en")["train"].shuffle()
vocab_size = 37000
input_sentence_size = 4000000


def get_iterable():
    for i, item in enumerate(train_data):
        if i == input_sentence_size // 2:
            break

        source_text = item["translation"]["de"]
        target_text = item["translation"]["en"]
        yield source_text
        yield target_text


data_iter = get_iterable()

# Initialize SentencePieceTrainer
spm.SentencePieceTrainer.Train(
    model_type="bpe",
    pad_id=3,
    model_prefix="shared_vocab_tokenizer",
    vocab_size=vocab_size,
    sentence_iterator=data_iter,
)
