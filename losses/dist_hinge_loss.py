import torch.nn as nn
import torch

class DistanceHingeLoss(nn.Module):

    def __init__(self, distance_threshold: float):
        super().__init__()
        self.distance_threshold = distance_threshold

    def forward(self, pos_a: torch.Tensor, pos_b: torch.Tensor = None):

        if pos_b is None:
            pairwise_distances = torch.cdist(pos_a, pos_a)
        else:
            pairwise_distances = torch.cdist(pos_a, pos_b)

        pairwise_loss = torch.max(-1*pairwise_distances+self.distance_threshold,  torch.zeros_like(pairwise_distances))

        if pos_b is None:
            loss = torch.triu(pairwise_loss, diagonal=1).sum()
        else:
            loss = pairwise_loss.sum()

        return loss

