from typing import Optional

import torch
from torch import nn


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
            output_token_idx: int,
            encoder_output: torch.Tensor,
            stack_size: int,
            model_dim: int,
            ff_hidden_layer_dim: int,
            num_attention_heads: int,
            key_dim: int,
            value_dim: int,
            p_dropout: float,
            eps_ls: float
    ):
        """
        Initialize the decoder.

        Args:
            stack_size (int): Number of encoder_blocks to stack.
            output_token_idx (int): The index of the token to be generated.
            encoder_output (torch.Tensor): The output of the encoder.
            model_dim (int): The input and output dimension of the model.
            ff_hidden_layer_dim (int): The dimension of the hidden layer in the feed-forward network.
            num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
            key_dim (int): The dimension of the key vectors in the multi-head attention mechanism.
            value_dim (int): The dimension of the value vectors in the multi-head attention mechanism.
            p_dropout (float): The dropout probability.
            eps_ls (float): The epsilon value for label smoothing.
        """
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            DecoderBlock(
                output_token_idx=output_token_idx,
                encoder_output=encoder_output,
                model_dim=model_dim,
                ff_hidden_layer_dim=ff_hidden_layer_dim,
                num_attention_heads=num_attention_heads,
                key_dim=key_dim,
                value_dim=value_dim,
                p_dropout=p_dropout,
                eps_ls=eps_ls
            ) for _ in range(stack_size)
        ])

    def forward(self, input):
        """
        Forward pass of the Decoder module.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, context_size, model_dim).

        """
        output = input
        for encoder_block in self.encoder_blocks:
            output = encoder_block(output)

        return output


class DecoderBlock(nn.Module):
    """
    Decoder block, which utilizes self-attention to decode the input sequence representation.
    """

    def __init__(
            self,
            output_token_idx: int,
            encoder_output: torch.Tensor,
            model_dim: int,
            ff_hidden_layer_dim: int,
            num_attention_heads: int,
            key_dim: int,
            value_dim: int,
            p_dropout: float,
            eps_ls: float
    ):
        """
        Initialize the decoder block.

        Args:
            output_token_idx (int): The index of the token to be generated.
            encoder_output (torch.Tensor): The output of the encoder.
            model_dim (int): The input and output dimension of the model.
            ff_hidden_layer_dim (int): The dimension of the hidden layer in the feed-forward network.
            num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
            key_dim (int): The dimension of the key vectors in the multi-head attention mechanism.
            value_dim (int): The dimension of the value vectors in the multi-head attention mechanism.
            p_dropout (float): The dropout probability.
            eps_ls (float): The epsilon value for label smoothing.
        """
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            mask_idx=output_token_idx
        )
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            encoder_output=encoder_output
        )
        self.batch_norm_1 = nn.BatchNorm1d(num_features=model_dim)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=model_dim)
        self.batch_norm_3 = nn.BatchNorm1d(num_features=model_dim)
        self.ffn = FeedForwardNetwork(model_dim=model_dim, hidden_layer_dim=ff_hidden_layer_dim)

    def forward(self, input):
        """
        Forward pass of the decoder block.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, context_size, model_dim).

        """
        x = input + self.masked_multi_head_attention(input)
        x = self.batch_norm_1(x)
        y = input + self.multi_head_attention(x)
        y = self.batch_norm_2(y)
        out = x + self.ffn(y)
        out = self.batch_norm_3(out)
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
            eps_ls: float
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
            eps_ls (float): The epsilon value for label smoothing.
        """
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                model_dim=model_dim,
                ff_hidden_layer_dim=ff_hidden_layer_dim,
                num_attention_heads=num_attention_heads,
                key_dim=key_dim,
                value_dim=value_dim,
                p_dropout=p_dropout,
                eps_ls=eps_ls
            ) for _ in range(stack_size)
        ])

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
            eps_ls: float
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
            eps_ls (float): The epsilon value for label smoothing.
        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_attention_heads,
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim
        )
        self.batch_norm_1 = nn.BatchNorm1d(num_features=model_dim)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=model_dim)
        self.ffn = FeedForwardNetwork(model_dim=model_dim, hidden_layer_dim=ff_hidden_layer_dim)

    def forward(self, input):
        """
        Forward pass of the encoder block.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, context_size, model_dim).

        """
        x = input + self.multi_head_attention(input)
        x = self.batch_norm_1(x)
        out = x + self.ffn(x)
        out = self.batch_norm_2(out)
        return out


class AttentionHead(nn.Module):
    """
    Attention head, which computes representations, based on the input elements and their relevance to each other.
    """

    def __init__(self, model_dim: int, key_dim: int, value_dim: int, mask_idx: Optional[int] = None,
                 encoder_output: Optional[torch.Tensor] = None):
        """
        Initialize the attention head.

        Args:
            model_dim (int): The input and output dimension of the model.
            key_dim (int):   The dimension of the keys in the attention mechanism.
            value_dim (int): The dimension of the values in the attention mechanism.

            Decoder arguments:
            mask_idx (Optional[int]): If provided, tokens corresponding to indices >= this value won't be attended to
                (default: None).
            encoder_output (Optional[torch.Tensor]): Output tensor from the encoder, if provided, it will be used for
                computing the key and value vectors in the attention mechanism (default: None).
        """
        super().__init__()
        self.key_dim = key_dim
        self.W_q = nn.Linear(in_features=model_dim, out_features=key_dim)
        self.W_k = nn.Linear(in_features=model_dim, out_features=key_dim)
        self.W_v = nn.Linear(in_features=model_dim, out_features=value_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.encoder_output = encoder_output
        self.mask_idx = mask_idx

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the AttentionLayer module.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: The attended values tensor of shape (batch_size, context_size, value_dim).
        """

        # Compute query (Q), key (K) and value (V) vectors.
        Q = self.W_q(input)
        K = self.W_k(self.encoder_output) if self.encoder_output else self.W_k(input)
        V = self.W_v(self.encoder_output) if self.encoder_output else self.W_v(input)

        # Compute attention weights, mask positions if mask_idx is given
        attention_weights = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(self.key_dim))
        if self.mask_idx:
            attention_weights[..., self.mask_idx:] = -torch.inf

        attention_weights = self.softmax(attention_weights)

        # Compute attention matrix
        attention = attention_weights @ V

        return attention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module, which concatenates and processes the output of multiple attention heads.
    """

    def __init__(self, num_heads: int, model_dim: int, key_dim: int, value_dim: int, mask_idx: Optional[int] = None,
                 encoder_output: Optional[torch.Tensor] = None):
        """
        Initialize the multi-head attention module.

        Args:
            num_heads (int): The number of attention heads.
            model_dim (int): The input and output dimension of the model.
            key_dim (int):   The dimension of the keys in the attention mechanism.
            value_dim (int): The dimension of the values in the attention mechanism.

            Decoder arguments:
            mask_idx (Optional[int]): If provided, tokens corresponding to indices >= this value won't be attended to
                (default: None).
            encoder_output (Optional[torch.Tensor]): Output tensor from the encoder, if provided, it will be used for
                computing the key and value vectors in the attention mechanism (default: None).
        """
        super().__init__()
        self.W_o = nn.Linear(in_features=key_dim * num_heads, out_features=model_dim)

        # Create attention heads
        self.attention_heads = nn.ModuleList([
            AttentionHead(model_dim=model_dim, key_dim=key_dim, value_dim=value_dim, mask_idx=mask_idx,
                          encoder_output=encoder_output) for _ in range(num_heads)
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
        concatenated_outputs = torch.cat(head_outputs, dim=-1)

        # Transform to model dimension
        output = self.W_o(concatenated_outputs)

        return output


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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, context_size, model_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, context_size, model_dim).

        """
        return self.linear2(self.relu(self.linear1(input)))
