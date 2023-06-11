from typing import Optional

import torch
from torch import nn, Tensor


def positional_encoding(context_size: int, embedding_dim: int) -> torch.Tensor:
    """
    Generate positional encoding for the input sequence.

    Args:
        context_size (int): The context size.
        embedding_dim (int): The embedding size.

    Returns:
        torch.Tensor: The positional encoding tensor of shape (context_size, model_dim).

    """
    positions = torch.arange(0, context_size)
    dim_indices = torch.arange(0, embedding_dim)

    # Compute the argument matrix for the sine and cosine function.
    def get_co_sine_arg(pos, dim_idx):
        return pos / (1e4 ** (dim_idx / embedding_dim))

    co_sine_args = get_co_sine_arg(positions.unsqueeze(1), dim_indices.unsqueeze(0))

    # Compute the sine or cosine, depending on whether the last dimension is even or odd.
    positional_encoding = torch.zeros((context_size, embedding_dim))
    positional_encoding[:, 0::2] = torch.sin(co_sine_args[:, 0::2])
    positional_encoding[:, 1::2] = torch.cos(co_sine_args[:, 1::2])

    return positional_encoding


class Decoder(nn.Module):
    """
    Decoder consisting of multiple stacked decoder-blocks.
    """

    def __init__(
        self,
        stack_size: int,
        model_dim: int,
        ff_hidden_layer_dim: int,
        num_attention_heads: int,
        key_dim: int,
        value_dim: int,
        p_dropout: float,
    ):
        """
        Initialize the decoder.

        Args:
            stack_size (int): Number of encoder_blocks to stack.
            model_dim (int): The input and output dimension of the model.
            ff_hidden_layer_dim (int): The dimension of the hidden layer in the feed-forward network.
            num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
            key_dim (int): The dimension of the key vectors in the multi-head attention mechanism.
            value_dim (int): The dimension of the value vectors in the multi-head attention mechanism.
            p_dropout (float): The dropout probability.
        """
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    model_dim=model_dim,
                    ff_hidden_layer_dim=ff_hidden_layer_dim,
                    num_attention_heads=num_attention_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    p_dropout=p_dropout,
                )
                for _ in range(stack_size)
            ]
        )

    def forward(self, input: Tensor, encoder_output: Tensor):
        """
        Forward pass of the Decoder module.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, context_size, model_dim).
            encoder_output (torch.Tensor): The output tensor from the encoder.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, context_size, model_dim).

        """
        output = input
        for decoder_block in self.encoder_blocks:
            output = decoder_block(output, encoder_output=encoder_output)

        return output


class DecoderBlock(nn.Module):
    """
    Decoder block, which utilizes self-attention to decode the input sequence representation.
    """

    def __init__(
        self,
        model_dim: int,
        ff_hidden_layer_dim: int,
        num_attention_heads: int,
        key_dim: int,
        value_dim: int,
        p_dropout: float,
    ):
        """
        Initialize the decoder block.

        Args:
            model_dim (int): The input and output dimension of the model.
            ff_hidden_layer_dim (int): The dimension of the hidden layer in the feed-forward network.
            num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
            key_dim (int): The dimension of the key vectors in the multi-head attention mechanism.
            value_dim (int): The dimension of the value vectors in the multi-head attention mechanism.
            p_dropout (float): The dropout probability.
        """
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
        )
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
        )
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)
        self.layer_norm_3 = nn.LayerNorm(model_dim)
        self.ffn = FeedForwardNetwork(
            model_dim=model_dim, hidden_layer_dim=ff_hidden_layer_dim
        )

    def forward(self, input: Tensor, encoder_output: Tensor):
        """
        Forward pass of the decoder block.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, context_size, model_dim).
            encoder_output (torch.Tensor): The output tensor from the encoder.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, context_size, model_dim).

        """
        x = input + self.masked_multi_head_attention(input, mask=True)
        x = self.layer_norm_1(x)
        y = input + self.multi_head_attention(x, encoder_output=encoder_output)
        y = self.layer_norm_2(y)
        out = x + self.ffn(y)
        out = self.layer_norm_3(out)
        return out


class Encoder(nn.Module):
    """
    Encoder consisting of multiple stacked encoder-blocks.
    """

    def __init__(
        self,
        stack_size: int,
        model_dim: int,
        ff_hidden_layer_dim: int,
        num_attention_heads: int,
        key_dim: int,
        value_dim: int,
        p_dropout: float,
    ):
        """
        Initialize the encoder.

        Args:
            stack_size (int): Number of encoder_blocks to stack.
            model_dim (int): The input and output dimension of the model.
            ff_hidden_layer_dim (int): The dimension of the hidden layer in the feed-forward network.
            num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
            key_dim (int): The dimension of the key vectors in the multi-head attention mechanism.
            value_dim (int): The dimension of the value vectors in the multi-head attention mechanism.
            p_dropout (float): The dropout probability.
        """
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    model_dim=model_dim,
                    ff_hidden_layer_dim=ff_hidden_layer_dim,
                    num_attention_heads=num_attention_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    p_dropout=p_dropout,
                )
                for _ in range(stack_size)
            ]
        )

    def forward(self, input):
        """
        Forward pass of the Encoder module.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, context_size, model_dim).

        """
        output = input
        for encoder_block in self.encoder_blocks:
            output = encoder_block(output)

        return output


class EncoderBlock(nn.Module):
    """
    Encoder block, which utilizes self-attention to encode the input sequence.
    """

    def __init__(
        self,
        model_dim: int,
        ff_hidden_layer_dim: int,
        num_attention_heads: int,
        key_dim: int,
        value_dim: int,
        p_dropout: float,
    ):
        """
        Initialize the encoder block.

        Args:
            model_dim (int): The input and output dimension of the model.
            ff_hidden_layer_dim (int): The dimension of the hidden layer in the feed-forward network.
            num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
            key_dim (int): The dimension of the key vectors in the multi-head attention mechanism.
            value_dim (int): The dimension of the value vectors in the multi-head attention mechanism.
            p_dropout (float): The dropout probability.
        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
        )
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)
        self.ffn = FeedForwardNetwork(
            model_dim=model_dim, hidden_layer_dim=ff_hidden_layer_dim
        )

    def forward(self, input):
        """
        Forward pass of the encoder block.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, context_size, model_dim).

        """
        x = input + self.multi_head_attention(input)
        x = self.layer_norm_1(x)
        out = x + self.ffn(x)
        out = self.layer_norm_2(out)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module, which concatenates and processes the output of multiple attention heads.
    """

    def __init__(self, num_heads: int, model_dim: int, key_dim: int, value_dim: int):
        """
        Initialize the multi-head attention module.

        Args:
            num_heads (int): The number of attention heads.
            model_dim (int): The input and output dimension of the model.
            key_dim (int):   The dimension of the keys in the attention mechanism.
            value_dim (int): The dimension of the values in the attention mechanism.

        """
        super().__init__()
        # Linear transformations for queries, keys and values
        self.q_linear = nn.Linear(in_features=model_dim, out_features=num_heads*key_dim)
        self.k_linear = nn.Linear(in_features=model_dim, out_features=num_heads*key_dim)
        self.v_linear = nn.Linear(in_features=model_dim, out_features=num_heads*value_dim)

        self.out_linear = nn.Linear(in_features=num_heads*value_dim, out_features=model_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

    def forward(
        self,
        input: Tensor,
        mask: Optional[bool] = False,
        encoder_output: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the multi-head attention module.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, context_size, model_dim).

            Decoder arguments:
            mask (Optional[bool]): If true, tokens are prevented to attend to subsequent positions of the sequence
                (default: False).
            encoder_output (Optional[torch.Tensor]): Output tensor from the encoder, if provided, it will be used for
                computing the key and value vectors in the attention mechanism (default: None).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, context_size, model_dim).
        """
        batch_size, context_size, model_dim = input.shape

        # Compute query (Q), key (K) and value (V) vectors
        Q = self.q_linear(input)
        K = self.k_linear(input) if encoder_output is None else self.k_linear(encoder_output)
        V = self.v_linear(input) if encoder_output is None else self.v_linear(encoder_output)

        # Reshape keys, queries, and values for multi-head attention
        Q = Q.view(batch_size, context_size, self.num_heads, self.key_dim).transpose(1, 2)
        K = K.view(batch_size, context_size, self.num_heads, self.key_dim).transpose(1, 2)
        V = V.view(batch_size, context_size, self.num_heads, self.value_dim).transpose(1, 2)

        # Compute attention
        attention = self.dot_product_attention(Q=Q, K=K, V=V, mask=mask)
        attention = attention.transpose(1, 2).reshape(batch_size, context_size, self.num_heads * self.value_dim)

        # Transform to model dimension
        output = self.out_linear(attention)

        return output

    def dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, mask: bool = False) -> Tensor:
        """
        Compute dot product attention.

        Args:
            Q (Tensor): Query tensor of shape (batch_size, num_heads, context_size, key_dim).
            K (Tensor): Key tensor of shape (batch_size, num_heads, context_size, key_dim).
            V (Tensor): Value tensor of shape (batch_size, num_heads, context_size, value_dim).
            mask (bool): Whether to apply masking to future positions. Default is False.

        Returns:
            Tensor: Tensor of shape (batch_size, num_heads, context_size, value_dim).
        """
        attention_weights = Q @ K.transpose(-2, -1) / (self.key_dim ** 0.5)

        # Compute attention weights, optionally, mask future positions
        # The i'th row corresponds to the attention values of the i'th token to all other tokens.
        if mask:
            attention_weights += torch.triu(torch.full(attention_weights.shape[-2:], -torch.inf), diagonal=1)

        attention_weights = self.softmax(attention_weights)
        attention = attention_weights @ V
        return attention


class FeedForwardNetwork(nn.Module):
    """
    A simple feed-forward network with one hidden layer.
    """

    def __init__(self, model_dim: int, hidden_layer_dim: int):
        """
        Initialize the feed-forward network.

        Args:
            model_dim (int): The input and output dimension of the model.
            hidden_layer_dim (int): The dimension of the hidden layer.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features=model_dim, out_features=hidden_layer_dim)
        self.linear2 = nn.Linear(in_features=hidden_layer_dim, out_features=model_dim)
        self.relu = nn.ReLU()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the network.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, context_size, model_dim).

        """
        return self.linear2(self.relu(self.linear1(input)))


if __name__ == "__main__":
    # Visualize the positional encoding.
    import matplotlib.pyplot as plt

    positional_encoding = positional_encoding(context_size=64, embedding_dim=256)

    plt.pcolormesh(positional_encoding)
    plt.xlabel("embedding dimension")
    plt.ylabel("token position")
    plt.show()
