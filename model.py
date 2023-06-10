import torch
from torch import nn
from transformer_components import Encoder, Decoder, positional_encoding


class Transformer(nn.Module):
    def __init__(self, context_size, source_vocab_size, target_vocab_size, model_dim, stack_size, ff_hidden_layer_dim,
                 num_attention_heads, key_dim, value_dim, p_dropout, pad_idx, bos_idx, eos_idx):
        super().__init__()
        self.positional_encoding = positional_encoding(context_size=context_size, embedding_dim=model_dim)
        self.src_embedding_layer = nn.Embedding(num_embeddings=source_vocab_size, embedding_dim=model_dim)
        self.tgt_embedding_layer = nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=model_dim)

        encoder_decoder_args = dict(stack_size=stack_size, model_dim=model_dim, ff_hidden_layer_dim=ff_hidden_layer_dim,
                                    num_attention_heads=num_attention_heads, key_dim=key_dim, value_dim=value_dim,
                                    p_dropout=p_dropout)
        self.encoder = Encoder(**encoder_decoder_args)
        self.decoder = Decoder(**encoder_decoder_args)

        self.linear = nn.Linear(in_features=context_size*model_dim, out_features=target_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.context_size = context_size
        self.model_dim = model_dim

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def encode_source(self, src_sequence):
        src_embedded = self.src_embedding_layer(src_sequence) + self.positional_encoding
        src_encoding = self.encoder(src_embedded)
        return src_encoding

    def forward(self, src_encoding, tgt_sequence, current_token_idx):
        tgt_embedded = self.tgt_embedding_layer(tgt_sequence) + self.positional_encoding
        tgt_decoded = self.decoder(input=tgt_embedded, current_token_idx=current_token_idx, encoder_output=src_encoding)

        tgt_decoded = tgt_decoded.reshape(-1, self.context_size*self.model_dim)
        output = self.linear(tgt_decoded)
        prb_next_token = self.softmax(output)

        return prb_next_token

    def generate(self, src_sequence):
        if src_sequence.dim() == 1:
            src_sequence = src_sequence.unsqueeze(0)

        batch_size = src_sequence.shape[0]
        src_encoding = self.encode_source(src_sequence)

        tgt_sequence = torch.full((batch_size, self.context_size), self.pad_idx)
        tgt_sequence[:, 0] = self.bos_idx

        for i in range(1, self.context_size):
            prb_next_token = self(src_encoding, tgt_sequence, i)
            next_token = torch.argmax(prb_next_token, dim=-1)
            tgt_sequence[:, i] = next_token

        return tgt_sequence


if __name__ == "__main__":
    from translation_dataset import get_english_german_translation_dataset
    from torch.utils.data import DataLoader
    import yaml

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Load config
    yaml_file = 'config.yaml'

    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Load the dataset
    dataset = get_english_german_translation_dataset()

    # Create a data loader for batching
    batch_size = 3
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    transformer = Transformer(**dataset.transformer_params, **config["transformer_params"])
    transformer.eval()

    # Iterate over batches of data
    print("Batched Data:")
    for batch in data_loader:
        input_batch, target_batch = batch
        print(f"Input Batch: {input_batch.shape}")
        print(f"Output Batch: {transformer.generate(input_batch).shape}")
        break
