import torch
from torch import nn
from transformer_components import Encoder, Decoder, positional_encoding


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        model_dim: int,
        stack_size: int,
        ff_hidden_layer_dim: int,
        num_attention_heads: int,
        key_dim: int,
        value_dim: int,
        p_dropout: float,
        target_bos_id: int,
        target_eos_id: int,
        source_pad_id: int
    ):
        """
        Initializes the Transformer model.

        Args:
            source_vocab_size (int): Size of the source vocabulary.
            target_vocab_size (int): Size of the target vocabulary.
            model_dim (int): Dimensionality of the model.
            stack_size (int): Number of encoder and decoder layers to stack.
            ff_hidden_layer_dim (int): Dimensionality of the feed-forward hidden layer.
            num_attention_heads (int): Number of attention heads.
            key_dim (int): Dimensionality of the key vectors in self-attention.
            value_dim (int): Dimensionality of the value vectors in self-attention.
            p_dropout (float): Dropout rate.
            target_bos_id (int): Index of the target beginning-of-sequence token.
            target_eos_id (int): Index of the target end-of-sequence token.
            source_pad_id (int): Index of the source padding token.
        """
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

        self.tgt_bos_id = target_bos_id
        self.tgt_eos_id = target_eos_id
        self.src_pad_id = source_pad_id

        self.model_dim = model_dim

        # Initialize the weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_source(
        self, src_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes the source sequence.

        Args:
            src_sequence (torch.Tensor): Source sequence tensor.

        Returns:
            torch.Tensor: Encoded source tensor.
        """
        src_key_padding_mask = src_sequence == self.src_pad_id

        src_embedded = self.src_embedding_layer(src_sequence) * self.model_dim**0.5
        src_embedded = src_embedded + positional_encoding(
            sequence_length=src_sequence.shape[-1], embedding_dim=self.model_dim
        )

        src_encoding = self.encoder(
            input_tensor=src_embedded, src_key_padding_mask=src_key_padding_mask
        )
        return src_encoding

    def forward(
        self,
        src_encoding: torch.Tensor,
        tgt_sequence: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the Transformer model.

        Args:
            src_encoding (torch.Tensor): Encoded source tensor.
            tgt_sequence (torch.Tensor): Target sequence tensor.
            src_key_padding_mask (torch.Tensor, optional): Mask indicating padding positions in the source sequence.

        Returns:
            torch.Tensor: Predicted probabilities of the next token.
        """
        tgt_embedded = self.tgt_embedding_layer(tgt_sequence) * self.model_dim**0.5
        tgt_embedded = tgt_embedded + positional_encoding(
            sequence_length=tgt_sequence.shape[-1], embedding_dim=self.model_dim
        )

        tgt_decoded = self.decoder(
            input_tensor=tgt_embedded,
            encoder_output=src_encoding,
            src_key_padding_mask=src_key_padding_mask,
        )

        logits = self.linear(tgt_decoded)

        return logits

    def generate(
        self,
        src_sequence: torch.Tensor,
        max_len: int = 128,
    ) -> torch.Tensor:
        """
        Generates target sequences given source sequences.

        Args:
            src_sequence (torch.Tensor): Source sequence tensor.
            max_len (int, optional): Maximum length of the generated sequences.

        Returns:
            torch.Tensor: Generated target sequences.
        """
        if src_sequence.ndim == 1:
            src_sequence = src_sequence.unsqueeze(0)

        batch_size, sequence_length = src_sequence.shape
        src_key_padding_mask = src_sequence == self.src_pad_id

        src_encoding = self.encode_source(src_sequence)
        tgt_sequence = torch.full((batch_size, 1), self.tgt_bos_id)

        for _ in range(max_len):
            logits = self(src_encoding, tgt_sequence, src_key_padding_mask)[
                :, -1
            ]
            next_token = torch.argmax(logits, dim=-1)
            tgt_sequence = torch.cat((tgt_sequence, next_token.unsqueeze(1)), dim=1)

            if torch.any(tgt_sequence == self.tgt_eos_id, dim=1).all():
                break

        return tgt_sequence


if __name__ == "__main__":
    from translation_datapipe import create_translation_datapipe_train
    from util import load_config, load_tokenizers, get_tokenizer_params

    # Load config
    config = load_config("config.yaml")

    # Load the tokenizers
    source_tokenizer, target_tokenizer = load_tokenizers(**config["tokenizer"])

    # Load the datapipe
    translation_datapipe = create_translation_datapipe_train(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        **config["datapipe_train"]
    )

    # Create the model
    transformer = Transformer(
        **get_tokenizer_params(source_tokenizer, target_tokenizer),
        **config["transformer_params"]
    )

    print(
        f"Number of trainable parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}"
    )

    # Iterate over batches of data
    print("Batched Data:")
    for batch in translation_datapipe:
        source_batch, target_batch = batch
        src_key_padding_mask = source_batch == source_tokenizer.pad_id()
        print(f"Input Batch: {source_batch}")
        print(f"Padding_mask: {src_key_padding_mask}")
        print(
            f"Output Batch: {transformer.generate(source_batch).shape}"
        )
        break
