import torch
from torch import nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Point-wise feed forward network.
    """

    def __init__(self, d_model: int, d_hidden: int = 2048, dropout_prob: float = 0.1):
        """
        Args:
            - d_model: the size of input for the first-layer of the FFN (equals to the dimension of linear projection for query, key and value matrices)
            - d_hidden: the hidden layer size of the second-layer
            - dropout_prob: probability of the dropout layer
        """
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

        self.dropout = nn.Dropout(p = dropout_prob)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x (torch.Tensor): input tensor in shape (batch_size, seq_len, d_model)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x