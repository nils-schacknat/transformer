import yaml
import sentencepiece as spm


def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_tokenizers(source_tokenizer_path, target_tokenizer_path):
    source_tokenizer = spm.SentencePieceProcessor(model_file=source_tokenizer_path)
    target_tokenizer = spm.SentencePieceProcessor(model_file=target_tokenizer_path)

    return source_tokenizer, target_tokenizer
