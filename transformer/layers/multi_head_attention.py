import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

def _scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the scaled dot product attention for given query, key and value matrices.

    Args:
        - q (torch.Tensor): The query matrix in shape (batch_size, n_head, q_length, d_k)
        - k (torch.Tensor): The key matrix in shape (batch_size, n_head, k_length, d_k)
        - v (torch.Tensor): The value matrix in shape (batch_size, n_head, v_length, d_v)
        - mask (torch.Tensor): The look ahead mask in shape (batch_size, n_head, ...) (default to None)
    
    Returns: (torch.Tensor, torch.Tensor) Attention values and attention weights.
        - Attention values in shape (batch_size, n_head, q_length, d_v)
        - attention weights in shape (batch_size, n_head, ...)
    """
    # Dimension of each key and query vector
    # d_k = torch.cast(k.shape[-1], dtype=torch.float32)
    d_k = torch.tensor(k.shape[-1], dtype=torch.float32)

    # Calculate attention weights
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)   # (..., q_length, k_lengh)
    if mask is not None:
        attention_scores += (mask * -1e30)
    attention_weights = F.softmax(attention_scores, dim=-1) 

    # Calculate attention values
    attention_values = torch.matmul(attention_weights, v) # (..., q_length, d_v)

    return attention_values, attention_weights



class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention layer as described in the paper.
    """

    def __init__(self, n_heads: int = 6, d_q: int = None, d_k: int = None, d_v: int = None, d_model: int = 512):
        """
        Args:
            - n_heads (int): Number of heads
            - d_q (int): The dimension of the query vector (default to None)
            - d_k (int): The dimension of the key vector (default to None)
            - d_v (int): The dimension of the value vector (default to None)
            - d_model (int): The dimension of linear projection for query, key and value matrices.
        """
        super(MultiHeadAttention, self).__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model

        if d_q is None:
            d_q = d_k = d_model
        if d_v is None:
            d_v = d_model

        # Linear Projectors
        self.w_q = nn.Linear(d_q, d_model)
        self.w_k = nn.Linear(d_k, d_model)
        self.w_v = nn.Linear(d_v, d_model)

        # Last feed forward layer
        self.w_out = nn.Linear(d_model, d_model)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            - q (torch.Tensor): The query matrix in shape (batch_size, q_length, d_q)
            - k (torch.Tensor): The key matrix in shape (batch_size, k_length, d_k)
            - v (torch.Tensor): The value matrix in shape (batch_size, v_length, d_v)
            - mask (torch.Tensor): The look ahead mask in shape (batch_size, ...) (default to None)
            
            where d_k, d_v are the dimension of each query, key and value vectors (d_q = d_k).

        Returns: (torch.Tensor, torch.Tensor) Output tensor and attention weights.
            - Output tensor in shape (batch_size, q_length, d_model)
            - Attention weights in shape ...
        """
        # Linear projection on query, key and value
        q_projected = self.w_q(q)   # (..., q_length, d_model)
        k_projected = self.w_k(k)   # (..., k_length, d_model)
        v_projected = self.w_v(v)   # (..., v_length, d_model)

        # Split each tensor to n_heads tensors
        q_heads = self.__split(q_projected)     # (..., n_head, q_length, d_model // n_heads)
        k_heads = self.__split(k_projected)     # (..., n_head, k_length, d_model // n_heads)
        v_heads = self.__split(v_projected)     # (..., n_head, v_length, d_model // n_heads)

        # Attention
        attention_values, attention_weights = _scaled_dot_product_attention(q_heads, k_heads, v_heads, mask = mask)

        # Merge heads
        out = self.__merge(attention_values)    # (..., q_length, d_model)

        # Feed forward
        out = self.w_out(out)       # (..., q_length, d_model)

        return out, attention_weights


    def __split(self, x: torch.tensor) -> torch.Tensor:
        """
        Split a tensor to n_heads tensors.

        Args:
            - x (torch.Tensor): Tensor in shape (..., length, d_model)

        Returns: (torch.Tensor) tensor in shape (..., n_heads, length, d_model // n_heads)
        """
        batch_size, length, d_model = x.size()

        assert d_model % self.n_heads == 0, "The number of heads must be divisible by the dimension of linear projection."
        
        d_tensor = d_model // self.n_heads

        new_x = x.view(batch_size, length, self.n_heads, d_tensor).transpose(1, 2)
        return new_x


    def __merge(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse function of __split.

        Args:
            - x (torch.Tensor): Tensor in shape (batch_size, n_heads, length, d_model // n_heads).

        Returns: (torch.Tensor) Tensor in shape (batch_size, length, d_model)
        """
        batch_size, head, length, d_tensor = x.size()
        d_model = head * d_tensor
        new_x = x.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return new_x