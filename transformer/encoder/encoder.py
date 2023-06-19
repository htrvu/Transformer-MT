import torch
from torch import nn
from torch.nn import Dropout, Embedding
from transformer.encoder.encoder_layer import EncoderLayer
from transformer.layers import LayerNorm
from transformer.helpers import calc_positional_encoding
from typing import Tuple, Dict
from torch.autograd import Variable

class Encoder(nn.Module):
    """
    Encoder for Transformer.
    """

    def __init__(self,
                 n_enc_layers: int = 6,
                 n_heads: int = 8,
                 vocab_size: int = 10000,
                 d_model: int = 512,
                 d_ffn_hidden: int = 2048,
                 dropout_prob: float = 0.1,
                 eps: float = 1e-6):
        '''
        Args:
            - n_enc_layers (int): Number of encoder layers (default to 6)
            - n_heads (int): Number of heads in Multi-head Attention layer (default to 8)
            - vocab_size (int): The size of vocabulary (default to 10000)
            - d_model (int): The dimension of linear projection for query, key and value matrices. This is also the dimension of
            embedding vector for words (default to 512)
            - d_ffn_hidden (int): The hidden layer size of the second-layer in FFN (default to 2048)
            - dropout_prob (float): The probability of the dropout layer (default to 0.1)
            - eps (float): The epsilon value for Layer Normalization (default to 1e-6)
        '''
        super(Encoder, self).__init__()
        
        self.n_enc_layers = n_enc_layers
        self.d_model = d_model

        self.word_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.enc_layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ffn_hidden, dropout_prob, eps) 
                                         for _ in range(n_enc_layers)])
        
        self.dropout = Dropout(dropout_prob)
        self.norm = LayerNorm(d_model, eps = eps)


    def forward(self, q: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        '''
        Args:
            - q (torch.Tensor): The query tensor in shape (batch_size, q_length).
            - is_train (bool): Whether the model is in training mode or not.
            - mask (torch.Tensor): The mask tensor in shape ...

        Returns: tuple[torch.Tensor, dict[str, torch.Tensor]] Output tensor and dictionary of self-attention weights.
            - The output tensor in shape (batch_size, q_length, d_model)
        '''
        q_length = q.size()[1]

        # Calculate word embedding and normalize
        q_embedded = self.word_embedding(q)     # (..., q_length, d_model)
        q_embedded *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Add positional encoding
        positional_encoding = calc_positional_encoding(last_pos=q_length, d_model=self.d_model).to(q.device)
        q_embedded += positional_encoding       # (..., q_length, d_model)

        # Dropout
        q_embedded = self.dropout(q_embedded)

        # Pass through encoder layers
        all_attn_weights = {}
        enc_out = q_embedded
        for i, enc_layer in enumerate(self.enc_layers):
            enc_out, attn_weights = enc_layer(enc_out, mask)      # (..., q_length, d_model)
            all_attn_weights[f'enc_layer_{i+1}_self_attn_weights'] = attn_weights

        # Normalize
        enc_out = self.norm(enc_out)
        
        return enc_out, all_attn_weights


# Test
if __name__ == '__main__':
    encoder = Encoder()
    inp = torch.randint(0, 10000, (64, 50))
    print(inp.shape)
    out = encoder(inp)
    print(out[0].shape)
    print(out[1]['enc_layer_1_self_attn_weights'].shape)