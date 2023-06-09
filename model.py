import torch
from torch import nn
from translation_dataset import get_english_german_translation_dataset
from torch.utils.data import DataLoader


class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embedding_size, context_size):
        super().__init__()
        self.src_embedding_layer = nn.Embedding(num_embeddings=source_vocab_size, embedding_dim=embedding_size)
        self.tgt_embedding_layer = nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=embedding_size)
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.linear = nn.Linear(in_features=embedding_size * context_size, out_features=target_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def encode_source(self, src_sequence):
        src_embedded = self.src_embedding_layer(src_sequence) + self.positional_encoding
        src_encoding = self.encoder(src_embedded)
        return src_encoding

    def forward(self, src_encoding, tgt_sequence):
        tgt_embedded = self.tgt_embedding_layer(tgt_sequence) + self.positional_encoding
        tgt_decoded = self.decoder(src_encoding, tgt_embedded)

        output = self.linear(tgt_decoded)
        prb_next_token = self.softmax(output)

        return prb_next_token


class EncoderBlock(nn.Module):
    def __init__(self, model_dim, ff_hidden_layer_dim, num_attention_heads, key_dim, value_dim, p_dropout, eps_ls):
        super().__init__()

    def forward(self, src_embedding):
        """
        src_embedding: n x c x h,
        n: batch_size, c: context_size, h: embedding_dim

        src_encoding:
        """


class AttentionHead(nn.Module):
    """
    Attention head, which computes representations, based on the input elements and their relevance to each other.
    """

    def __init__(self, model_dim: int, key_dim: int, value_dim: int):
        """
        Initialize the attention head.

        Args:
            model_dim (int): The output dimension of all model sub-layers.
            key_dim (int):   The dimension of the keys in the attention mechanism.
            value_dim (int): The dimension of the values in the attention mechanism.
        """
        super().__init__()
        self.key_dim = key_dim
        self.W_q = nn.Linear(in_features=model_dim, out_features=key_dim)
        self.W_k = nn.Linear(in_features=model_dim, out_features=key_dim)
        self.W_v = nn.Linear(in_features=model_dim, out_features=value_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the AttentionLayer module.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: The attended values tensor of shape (batch_size, context_size, value_dim).
        """

        # Compute query (Q), key (K) and value (V) vectors.
        Q = self.W_q(input)  # Shape: (batch_size, context_size, key_dim)
        K = self.W_k(input)  # Shape: (batch_size, context_size, key_dim)
        V = self.W_v(input)  # Shape: (batch_size, context_size, value_dim)

        attention_weights = self.softmax(Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(self.key_dim)))  # Shape: (batch_size, context_size, context_size)
        attention = attention_weights @ V  # Shape: (batch_size, context_size, value_dim)

        return attention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module, to jointly attend to different parts of the input representation multiple times.
    """

    def __init__(self, num_heads: int, model_dim: int, key_dim: int, value_dim: int):
        """
        Initialize the multi-head attention module.

        Args:
            num_heads (int): The number of attention heads.
            model_dim (int): The output dimension of all model sub-layers.
            key_dim (int):   The dimension of the keys in the attention mechanism.
            value_dim (int): The dimension of the values in the attention mechanism.
        """
        super().__init__()
        self.W_o = nn.Linear(in_features=key_dim * num_heads, out_features=model_dim)

        # Create attention heads
        self.attention_heads = nn.ModuleList([
            AttentionHead(model_dim=model_dim, key_dim=key_dim, value_dim=value_dim) for _ in range(num_heads)
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention module.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, context_size, model_dim).
        """
        # Apply each attention head and concatenate the outputs
        head_outputs = [attention_head(input) for attention_head in self.attention_heads]

        # Concatenate the head outputs along the last dimension
        concatenated_outputs = torch.cat(head_outputs, dim=-1)  # Shape: (batch_size, context_size, value_dim*num_heads)

        # Transform to model dimension
        output = self.W_o(concatenated_outputs)  # Shape: (batch_size, context_size, model_dim)

        return output


"""
Input:
inputs: context_size
embedded_inputs: context_size x model_dim           (model_dim = embedding_dim)

Attention:
queries: context_size x key_dim = (context_size x model_dim) * (model_dim x key_dim)
keys: context_size x key_dim = (context_size x model_dim) * (model_dim x key_dim)
values: context_size x value_dim = (context_size x model_dim) * (model_dim x value_dim)

[attention = softmax(QK^T/sqrt(key_dim)) * V]
attention = softmax(context_size x context_size) * (context_size x value_dim) (softmax along rows)
attention = context_size x value_dim

MultiHeadAttention:
concatenate h different attention results: context_size x (value_dim * h)
multi_headed_attention = context_size x model_dim = (context_size x (value_dim * h)) * ((value_dim * h) x model_dim)
"""


if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    context_size = 64

    # Load the dataset
    dataset = get_english_german_translation_dataset(context_size=context_size)

    # Create a data loader for batching
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    transformer = Transformer(source_vocab_size=dataset.source_vocab_size, target_vocab_size=dataset.target_vocab_size,
                              embedding_size=16, context_size=context_size)
    transformer.eval()

    # Iterate over batches of data
    print("Batched Data:")
    for batch in data_loader:
        input_batch, target_batch = batch
        src_encoding = transformer.encode_source(input_batch)
        print(f"Input Batch: {input_batch.shape}")
        print(f"Output Batch: {transformer(input_batch).shape}")
        break
