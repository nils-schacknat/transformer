import torch
from torch import nn
from transformer_components import Encoder, Decoder, positional_encoding


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        model_dim,
        stack_size,
        ff_hidden_layer_dim,
        num_attention_heads,
        key_dim,
        value_dim,
        p_dropout,
        bos_idx,
        eos_idx,
    ):
        super().__init__()
        self.src_embedding_layer = nn.Embedding(
            num_embeddings=source_vocab_size, embedding_dim=model_dim
        )
        self.tgt_embedding_layer = nn.Embedding(
            num_embeddings=target_vocab_size, embedding_dim=model_dim
        )

        encoder_decoder_args = dict(
            stack_size=stack_size,
            model_dim=model_dim,
            ff_hidden_layer_dim=ff_hidden_layer_dim,
            num_attention_heads=num_attention_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            p_dropout=p_dropout,
        )
        self.encoder = Encoder(**encoder_decoder_args)
        self.decoder = Decoder(**encoder_decoder_args)

        self.linear = nn.Linear(in_features=model_dim, out_features=target_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.model_dim = model_dim

    def encode_source(self, src_sequence, src_key_padding_mask=None):
        src_embedded = self.src_embedding_layer(src_sequence) * self.model_dim**.5 + positional_encoding(
            sequence_length=src_sequence.shape[-1], embedding_dim=self.model_dim,
        )
        src_encoding = self.encoder(input=src_embedded, src_key_padding_mask=src_key_padding_mask)
        return src_encoding

    def forward(self, src_encoding, tgt_sequence, src_key_padding_mask=None):
        tgt_embedded = self.tgt_embedding_layer(tgt_sequence) * self.model_dim**.5 + positional_encoding(
            sequence_length=tgt_sequence.shape[-1], embedding_dim=self.model_dim
        )
        tgt_decoded = self.decoder(input=tgt_embedded, encoder_output=src_encoding, src_key_padding_mask=src_key_padding_mask)

        output = self.linear(tgt_decoded)
        prb_next_token = self.softmax(output)

        return prb_next_token

    def generate(self, src_sequence, src_key_padding_mask=None, max_len=100):
        if src_sequence.ndim == 1:
            src_sequence = src_sequence.unsqueeze(0)

        batch_size, sequence_length = src_sequence.shape

        src_encoding = self.encode_source(src_sequence, src_key_padding_mask)
        tgt_sequence = torch.full((batch_size, 1), self.bos_idx)

        for _ in range(max_len):
            prb_next_token = self(src_encoding, tgt_sequence, src_key_padding_mask)[:, -1]
            next_token = torch.argmax(prb_next_token, dim=-1)
            tgt_sequence = torch.cat((tgt_sequence, next_token.unsqueeze(1)), dim=1)

            if torch.any(tgt_sequence == self.eos_idx, dim=1).all():
                break

        return tgt_sequence


if __name__ == "__main__":
    from translation_dataset import load_german_english_translation_dataset
    from torch.utils.data import DataLoader
    import yaml

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # Load config
    yaml_file = "config.yaml"

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    # Load the dataset
    dataset = load_german_english_translation_dataset()

    # Create a data loader for batching
    batch_size = 3
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    transformer = Transformer(
        **dataset.transformer_params, **config["transformer_params"]
    )
    transformer.eval()

    print(
        f"Number of trainable parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}"
    )

    # Iterate over batches of data
    print("Batched Data:")
    for batch in data_loader:
        input_batch, target_batch = batch
        src_key_padding_mask = torch.rand(input_batch.shape) < 0.5
        print(f"Input Batch: {input_batch.shape}")
        print(f"Output Batch: {transformer.generate(input_batch, src_key_padding_mask).shape}")
        break
