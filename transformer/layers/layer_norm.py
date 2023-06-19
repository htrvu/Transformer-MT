import torch
from torch import nn

class LayerNorm(nn.Module):
    '''
    Layer Normalization module for Transformer.
    '''
    def __init__(self, d_model: int = 512, eps: float = 1e-16):
        '''
        Args:
            - d_model (int): The dimension of linear projection for query, key and value matrices. This is also the dimension of
            embedding vector for words (default to 512)

            - eps (float): The epsilon value for Layer Normalization (default to 1e-12)
        '''
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            - x (torch.Tensor): The input tensor in shape (batch_size, ..., d_model)
        
        Returns: (torch.Tensor) The output tensor in shape (batch_size, ..., d_model)
        '''
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out