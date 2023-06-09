import torch
from torch import nn
from torch.nn import Dropout, Embedding
from transformer.decoder.decoder_layer import DecoderLayer
from transformer.helpers import calc_positional_encoding
from typing import Tuple, Dict

class Decoder(nn.Module):
    """
    Decoder for Transformer.
    """

    def __init__(self,
                 n_dec_layers: int = 6,
                 n_heads: int = 8,
                 vocab_size: int = 10000,
                 d_model: int = 512,
                 d_ffn_hidden: int = 2048,
                 dropout_prob: float = 0.1,
                 eps: float = 0.1):
        '''
        Args:
            - n_dec_layers (int): Number of decoder layers (default to 6)
            - n_heads (int): Number of heads in Multi-head Attention layer (default to 8)
            - vocab_size (int): The size of vocabulary (default to 10000)
            - d_model (int): The dimension of linear projection for query, key and value matrices. This is also the dimension of
            embedding vector for words (default to 512)
            - d_ffn_hidden (int): The hidden layer size of the second-layer in FFN (default to 2048)
            - dropout_prob (float): The probability of the dropout layer (default to 0.1)
            - eps (float): The epsilon value for Layer Normalization (default to 0.1)
        '''
        super(Decoder, self).__init__()
        
        self.d_model = d_model

        self.dec_layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ffn_hidden, dropout_prob, eps) 
                                         for _ in range(n_dec_layers)])
        
        self.word_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.dropout = Dropout(dropout_prob)


    # def forward(self, q: torch.Tensor, enc_output: torch.Tensor, look_a = None) -> torch.Tensor:
    def forward(self, 
                q: torch.Tensor, 
                enc_output: torch.Tensor, 
                look_ahead_mask: torch.Tensor = None, 
                padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor]:
        '''
        Args:
            - q (torch.Tensor): The query tensor in shape (batch_size, q_length).
            - enc_output (torch.Tensor): The output tensor from the encoder in shape (batch_size, k_length, d_model).
            - look_ahead_mask (torch.Tensor): The look ahead mask for decoder input in shape (batch_size, 1, q_length, q_length)
            - padding_mask (torch.Tensor): The padding mask for cross-attention in shape (batch_size, 1, 1, k_length)

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
        dec_out = q_embedded
        for i, dec_layer in enumerate(self.dec_layers):
            dec_out, self_attn_weights, cross_attn_weights = dec_layer(dec_out, enc_output, look_ahead_mask, padding_mask)      # (..., q_length, d_model)
            all_attn_weights[f'decoder_layer_{i+1}_self_attn_weights']: self_attn_weights
            all_attn_weights[f'decoder_layer_{i+1}_cross_attn_weights']: cross_attn_weights

        return dec_out, all_attn_weights


# Test
if __name__ == '__main__':
    from transformer.encoder import Encoder
    encoder = Encoder()
    decoder = Decoder()
    inp = torch.randint(0, 10000, (64, 50))
    enc_output = encoder(inp)[0]

    dec_output = decoder(inp, enc_output)[0]
    print(inp.shape)
    print(enc_output.shape)
    print(dec_output.shape)