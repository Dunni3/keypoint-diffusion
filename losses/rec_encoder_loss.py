from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import dgl
import ot
from typing import List
import numpy as np
from losses.dist_hinge_loss import DistanceHingeLoss

# this function is taken from equibind
def compute_ot_emd(cost_mat, device):
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    a = np.ones([cost_mat.shape[0]]) / cost_mat.shape[0]
    b = np.ones([cost_mat.shape[1]]) / cost_mat.shape[1]
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached

class ReceptorEncoderLoss(nn.Module):

    def __init__(self, loss_type='optimal_transport', hinge_threshold: float = 4):
        super().__init__()

        self.loss_type = loss_type

        if self.loss_type not in ['optimal_transport', 'gaussian_repulsion', 'hinge', 'none']:
            raise ValueError
        
        if self.loss_type == 'hinge':
            self._hinge_loss = DistanceHingeLoss(distance_threshold=hinge_threshold)

    def forward(self, keypoint_positions: List[torch.Tensor] = None, batched_rec_graphs: dgl.DGLGraph = None):

        if self.loss_type == "optimal_transport":
            return self.compute_ot_loss(keypoint_positions=keypoint_positions, batched_rec_graphs=batched_rec_graphs)
        elif self.loss_type == 'gaussian_repulsion':
            return self.compute_repulsion_loss(keypoint_positions=keypoint_positions)
        elif self.loss_type == 'hinge':
            return self.compute_hinge_loss(keypoint_positions=keypoint_positions)
        elif self.loss_type == 'none':
            device = keypoint_positions[0].device
            dtype = keypoint_positions[0].dtype
            return torch.tensor(0.0, device=device, dtype=dtype)

    def compute_ot_loss(self, keypoint_positions: List[torch.Tensor], batched_rec_graphs: dgl.DGLGraph):
        ot_loss = 0

        unbatched_graphs = dgl.unbatch(batched_rec_graphs)

        for kp_pos, rec_graph in zip(keypoint_positions, unbatched_graphs):
            # compute cost matrix
            cost_mat = torch.square(torch.cdist(kp_pos, rec_graph.ndata['x_0']))
            # compute OT distance
            # TODO: what should device be? not really sure how devices work in pytorch...need to figure that out
            ot_dist, _ = compute_ot_emd(cost_mat, device=cost_mat.device)
            ot_loss += ot_dist
        
        ot_loss = ot_loss / len(unbatched_graphs)

        return ot_loss

    def compute_repulsion_loss(self, keypoint_positions: List[torch.Tensor]):

        batch_size = len(keypoint_positions)
        n_keypoints = keypoint_positions[0].shape[0]
        n_pairs = n_keypoints*(n_keypoints - 1)/2

        repulsion_loss = 0
        for kp_pos in keypoint_positions:
            pairwise_distances = torch.cdist(kp_pos, kp_pos)
            pairwise_repulsion = torch.exp(-1*pairwise_distances)

            # compute loss as the sum of off-diagonal pairwise distances for only the top half
            repulsion_loss += torch.triu(pairwise_repulsion, diagonal=1).sum()

        # normalize by batch size and number of keypoint pairs
        repulsion_loss = repulsion_loss / batch_size / n_pairs
        
        return repulsion_loss
    
    def compute_hinge_loss(self, keypoint_positions: List[torch.Tensor]):

        batch_size = len(keypoint_positions)
        n_keypoints = keypoint_positions[0].shape[0]
        n_pairs = n_keypoints*(n_keypoints - 1)/2

        loss = 0
        for kp_pos in keypoint_positions:
            loss += self._hinge_loss(kp_pos)

        loss = loss / batch_size / n_pairs
        return loss