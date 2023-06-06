import torch
from torch import nn
from transformer.layers import *
from typing import Tuple


class DecoderLayer(nn.Module):
    """
    Decoder layer (or block) for Transformer.
    """

    def __init__(self, 
                 n_heads: int, 
                 d_model: int, 
                 d_ffn_hidden: int, 
                 dropout_prob: float = 0.1, 
                 eps: float = 0.1):
        """
        Args:
            - n_heads (int): The number of heads in Multi-head Attention layer.
            - d_model (int): The dimension of linear projection for query, key and value matrices. This is also the dimension of 
            embedding vector for words.
            - d_ffn_hidden (int): The hidden layer size of the second-layer in FFN.
            - dropout_prob (float): The probability of the dropout layer.
            - eps (float): The epsilon value for Layer Normalization.
        """
        super(DecoderLayer, self).__init__()

        self.masked_mha = MultiHeadAttention(n_heads = n_heads, d_model = d_model)
        self.mha = MultiHeadAttention(n_heads = n_heads, d_model = d_model)

        self.ffn = FeedForward(d_model = d_model, d_hidden = d_ffn_hidden, dropout_prob = dropout_prob)

        self.norm1 = nn.LayerNorm(d_model, eps = eps)
        self.dropout1 = nn.Dropout(p = dropout_prob)

        self.norm2 = nn.LayerNorm(d_model, eps = eps)
        self.dropout2 = nn.Dropout(p = dropout_prob)

        self.norm3 = nn.LayerNorm(d_model, eps = eps)
        self.dropout3 = nn.Dropout(p = dropout_prob)


    def forward(self, 
                q: torch.Tensor, 
                enc_output: torch.Tensor, 
                look_ahead_mask: torch.Tensor = None, 
                padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        Args:
            - q (torch.Tensor): The query tensor in shape (batch_size, q_length, d_model).
            - enc_output (torch.Tensor): The output tensor from the encoder in shape (batch_size, k_length, d_model).
            - look_ahead_mask (torch.Tensor): The look ahead mask for decoder input in shape (batch_size, 1, q_length, q_length)
            - padding_mask (torch.Tensor): The padding mask for cross-attention in shape (batch_size, 1, 1, k_length)
        
        Returns: tuple[torch.Tensor] Output tensor, self-attention weights and cross-attention weights.
            - The output tensor in shape (batch_size, q_length, d_model).
        """
        # Masked self-attention
        masked_mha_out, self_attn_weights = self.masked_mha(q = q, k = q, v = q, mask = look_ahead_mask)

        # Skip connection and layer norm
        q = self.norm1(q + self.dropout1(masked_mha_out))

        # Cross-attention with encoder output
        mha_out, cross_attn_weights = self.mha(q = q, k = enc_output, v = enc_output, mask = padding_mask)

        # Skip connection and layer norm
        q = self.norm2(q + self.dropout2(mha_out))

        # Feed forward
        ffn_out = self.ffn(q)

        # Skip connection and layer norm
        out = self.norm3(q + self.dropout3(ffn_out))

        return out, self_attn_weights, cross_attn_weights