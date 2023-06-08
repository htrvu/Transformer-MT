import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss function
    """

    def __init__(
        self, classes: int, padding_idx: int, smoothing: float = 0.0, dim: int = -1
    ) -> None:
        """
        Args:
            - classes (int): Number of classes
            - padding_idx (int): Index of padding
            - smoothing (float): Smoothing value
            - dim (int): Dimension to apply smoothing
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward propagation

        Args:
            - pred (torch.Tensor): Prediction tensor
            - target (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Loss value
        """
        pred = pred.log_softmax(dim=self.dim)

        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
