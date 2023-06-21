from collections import deque
from typing import Optional

import torch
from torch import nn, Tensor

# If this queue is initialized, it stores the attention weights for visualization
attention_weights_queue = None


def initialize_attention_weights_queue(size: int) -> None:
    """
    Initialize a queue to store the attention weights for visualization

    Args:
        size (int): The maximum length of the queue.

    """
    global attention_weights_queue
    attention_weights_queue = deque(maxlen=size)


def positional_encoding(sequence_length: int, embedding_dim: int) -> torch.Tensor:
    """
    Generate positional encoding for the input sequence.

    Args:
        sequence_length (int): The sequence length.
        embedding_dim (int): The embedding size.

    Returns:
        torch.Tensor: The positional encoding tensor of shape (sequence_length, model_dim).

    """
    positions = torch.arange(0, sequence_length)
    dim_indices = torch.arange(0, embedding_dim)

    # Compute the argument matrix for the sine and cosine function.
    def get_co_sine_arg(pos, dim_idx):
        return pos / (1e4 ** (dim_idx / embedding_dim))

    co_sine_args = get_co_sine_arg(positions.unsqueeze(1), dim_indices.unsqueeze(0))

    # Compute the sine or cosine, depending on whether the last dimension is even or odd.
    positional_encoding = torch.zeros((sequence_length, embedding_dim))
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

    def forward(
        self,
        input_tensor: Tensor,
        encoder_output: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Forward pass of the Decoder module.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, model_dim).
            encoder_output (torch.Tensor): The output_tensor tensor from the encoder.
            src_key_padding_mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, sequence_length) indicating
                the padding positions in the source sequence, padding positions should be set to 'True' (default: None).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, model_dim).

        """
        output_tensor = input_tensor
        for decoder_block in self.decoder_blocks:
            output_tensor = decoder_block(
                input_tensor=output_tensor,
                encoder_output=encoder_output,
                src_key_padding_mask=src_key_padding_mask,
            )

        return output_tensor


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
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self,
        input_tensor: Tensor,
        encoder_output: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Forward pass of the decoder block.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, model_dim).
            encoder_output (torch.Tensor): The output tensor from the encoder.
            src_key_padding_mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, sequence_length) indicating
                the padding positions in the source sequence, padding positions should be set to 'True' (default: None).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, model_dim).

        """
        x = input_tensor + self.dropout(
            self.masked_multi_head_attention(input_tensor, mask_future_positions=True)
        )
        x = self.layer_norm_1(x)

        x += self.dropout(
            self.multi_head_attention(
                x,
                encoder_output=encoder_output,
                src_key_padding_mask=src_key_padding_mask,
            )
        )
        x = self.layer_norm_2(x)

        x += self.dropout(self.ffn(x))
        x = self.layer_norm_3(x)
        return x


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

    def forward(
        self, input_tensor: Tensor, src_key_padding_mask: Optional[Tensor] = None
    ):
        """
        Forward pass of the Encoder module.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, model_dim).
            src_key_padding_mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, sequence_length) indicating
                the padding positions in the source sequence, padding positions should be set to 'True' (default: None).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, model_dim).

        """
        output_tensor = input_tensor
        for encoder_block in self.encoder_blocks:
            output_tensor = encoder_block(
                output_tensor, src_key_padding_mask=src_key_padding_mask
            )

        return output_tensor


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
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self, input_tensor: Tensor, src_key_padding_mask: Optional[Tensor] = None
    ):
        """
        Forward pass of the encoder block.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, model_dim).
            src_key_padding_mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, sequence_length) indicating
                the padding positions in the source sequence, padding positions should be set to 'True' (default: None).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, model_dim).

        """
        x = input_tensor + self.dropout(
            self.multi_head_attention(
                input_tensor=input_tensor, src_key_padding_mask=src_key_padding_mask
            )
        )
        x = self.layer_norm_1(x)

        x += self.dropout(self.ffn(x))
        x = self.layer_norm_2(x)
        return x


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
        self.q_linear = nn.Linear(
            in_features=model_dim, out_features=num_heads * key_dim
        )
        self.k_linear = nn.Linear(
            in_features=model_dim, out_features=num_heads * key_dim
        )
        self.v_linear = nn.Linear(
            in_features=model_dim, out_features=num_heads * value_dim
        )

        self.out_linear = nn.Linear(
            in_features=num_heads * value_dim, out_features=model_dim
        )
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

    def forward(
        self,
        input_tensor: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        mask_future_positions: Optional[bool] = False,
        encoder_output: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the multi-head attention module.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, sequence_length, model_dim).
            src_key_padding_mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, sequence_length) indicating
                the padding positions in the source sequence, padding positions should be set to 'True' (default: None).

            Decoder arguments:
            mask_future_positions (Optional[bool]): If true, tokens are prevented from attending to subsequent positions
                of the sequence (default: False).
            encoder_output (Optional[torch.Tensor]): Output tensor from the encoder, if provided, it will be used for
                computing the key and value vectors in the attention mechanism (default: None).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, model_dim).
        """
        # The sequence length of the input
        batch_size, sequence_length, model_dim = input_tensor.shape
        # The sequence length of the keys and values
        key_value_sequence_length = (
            encoder_output.shape[1] if encoder_output is not None else sequence_length
        )

        # Compute query (Q), key (K) and value (V) vectors
        Q = self.q_linear(input_tensor)
        K = (
            self.k_linear(input_tensor)
            if encoder_output is None
            else self.k_linear(encoder_output)
        )
        V = (
            self.v_linear(input_tensor)
            if encoder_output is None
            else self.v_linear(encoder_output)
        )

        # Reshape keys, queries, and values for multi-head attention
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.key_dim).transpose(
            1, 2
        )
        K = K.view(
            batch_size, key_value_sequence_length, self.num_heads, self.key_dim
        ).transpose(1, 2)
        V = V.view(
            batch_size, key_value_sequence_length, self.num_heads, self.value_dim
        ).transpose(1, 2)

        # Compute attention
        attention = self.dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            mask_future_positions=mask_future_positions,
            src_key_padding_mask=src_key_padding_mask,
        )

        attention = attention.transpose(1, 2).reshape(
            batch_size, sequence_length, self.num_heads * self.value_dim
        )

        # Transform to model dimension
        output_tensor = self.out_linear(attention)

        return output_tensor

    def dot_product_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask_future_positions: bool = False,
        src_key_padding_mask: Tensor = None,
    ) -> Tensor:
        """
        Compute dot product attention.

        Args:
            Q (Tensor): Query tensor of shape (batch_size, num_heads, sequence_length, key_dim).
            K (Tensor): Key tensor of shape (batch_size, num_heads, key_value_sequence_length, key_dim).
            V (Tensor): Value tensor of shape (batch_size, num_heads, key_value_sequence_length, value_dim).
            mask_future_positions (Optional[bool]): If true, tokens are prevented from attending to subsequent positions
                of the sequence (default: False).
            src_key_padding_mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, sequence_length) indicating
                the padding positions in the source sequence, padding positions should be set to 'True' (default: None).

        Returns:
            Tensor: Tensor of shape (batch_size, num_heads, sequence_length, value_dim).
        """
        attention_weights = Q @ K.transpose(-2, -1) / (self.key_dim**0.5)

        # Compute attention weights, optionally, mask future/padding positions
        # The i'th row corresponds to the attention values of the i'th token to all other tokens.
        # Dim 0 corresponds to queries and dim 1 to keys.
        if mask_future_positions:
            attention_weights += torch.triu(
                torch.full(attention_weights.shape[-2:], -torch.inf), diagonal=1
            )

        if src_key_padding_mask is not None:
            attention_weights += (
                (torch.where(src_key_padding_mask, -torch.inf, 0))
                .unsqueeze(1)
                .unsqueeze(1)
            )

        attention_weights = self.softmax(attention_weights)

        # Store attention weights for visualization
        if attention_weights_queue is not None:
            attention_weights_queue.append(attention_weights)

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

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass of the network.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, sequence_length, model_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, model_dim).

        """
        return self.linear2(self.relu(self.linear1(input_tensor)))


if __name__ == "__main__":
    # Visualize the positional encoding.
    import matplotlib.pyplot as plt

    positional_encoding = positional_encoding(sequence_length=64, embedding_dim=256)

    plt.pcolormesh(positional_encoding)
    plt.xlabel("embedding dimension")
    plt.ylabel("token position")
    plt.show()
