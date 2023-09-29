from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import dgl
import ot
from typing import List
import numpy as np
from losses.dist_hinge_loss import DistanceHingeLoss
from geomloss import SamplesLoss

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

    def __init__(self, loss_type='optimal_transport', use_interface_points: bool = False, hinge_threshold: float = 4, **kwargs):
        super().__init__()

        self.loss_type = loss_type
        self.use_interface_points = use_interface_points
        print('Rec Encoder Loss Kwargs: ', kwargs)

        if self.loss_type not in ['optimal_transport', 'gaussian_repulsion', 'hinge', 'geom', 'none']:
            raise ValueError

        if self.loss_type == 'hinge':
            self._hinge_loss = DistanceHingeLoss(distance_threshold=hinge_threshold)
        
        if self.loss_type == 'geom':
            self.geomloss_type = kwargs['geomloss_kwargs']['geomloss_type']

    def forward(self, batched_complex_graphs: dgl.DGLGraph = None, interface_points: List[torch.Tensor] = None):

        if self.loss_type == "optimal_transport" and not self.use_interface_points:
            return self.compute_ot_loss(batched_complex_graphs)
        elif self.loss_type == 'optimal_transport' and self.use_interface_points:
            return self.compute_interface_point_loss(batched_complex_graphs, interface_points)
        elif self.loss_type == 'geom' and self.use_interface_points:
            return self.compute_geom_loss(batched_complex_graphs, interface_points)
        elif self.loss_type == 'gaussian_repulsion':
            return self.compute_repulsion_loss(batched_complex_graphs)
        elif self.loss_type == 'hinge':
            return self.compute_hinge_loss(batched_complex_graphs)
        elif self.loss_type == 'none':
            device = batched_complex_graphs.device
            dtype = batched_complex_graphs.nodes["rec"].data["x_0"].dtype
            return torch.tensor(0.0, device=device, dtype=dtype)

    def compute_ot_loss(self, batched_complex_graphs: dgl.DGLGraph):

        ot_loss = 0

        unbatched_graphs = dgl.unbatch(batched_complex_graphs)

        # for kp_pos, rec_graph in zip(keypoint_positions, unbatched_graphs):
        for g in unbatched_graphs:

            kp_pos = g.nodes["kp"].data["x_0"]
            rec_atom_pos = g.nodes["rec"].data["x_0"]

            # compute cost matrix
            cost_mat = torch.square(torch.cdist(kp_pos, rec_atom_pos))
            # compute OT distance
            # TODO: what should device be? not really sure how devices work in pytorch...need to figure that out
            ot_dist, _ = compute_ot_emd(cost_mat, device=cost_mat.device)
            ot_loss += ot_dist
        
        ot_loss = ot_loss / len(unbatched_graphs)
        return ot_loss
    
    def compute_interface_point_loss(self, batched_complex_graphs, interface_points):

        keypoint_positions = [ g.nodes["kp"].data["x_0"] for g in dgl.unbatch(batched_complex_graphs) ]

        ot_loss = 0
        for kp_pos, if_points in zip(keypoint_positions, interface_points):
            cost_mat = torch.square(torch.cdist(kp_pos, if_points))
            ot_dist, _ = compute_ot_emd(cost_mat, device=cost_mat.device)
            ot_loss += ot_dist
        
        ot_loss = ot_loss / len(interface_points)
        return ot_loss
    
    def compute_geom_loss(self, batched_complex_graphs, interface_points):
        print("Geomloss Type: ", self.geomloss_type)

        keypoint_positions = [ g.nodes["kp"].data["x_0"] for g in dgl.unbatch(batched_complex_graphs) ]
        calc_geomloss = SamplesLoss(loss=self.geomloss_type, p=2, blur=.05)
        total_geomloss = 0
        for i in range(len(keypoint_positions)):
            total_geomloss += calc_geomloss(keypoint_positions[i], interface_points[i])

        return total_geomloss / len(interface_points)

    def compute_repulsion_loss(self, batched_complex_graphs: dgl.DGLHeteroGraph):

        raise NotImplementedError

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
    
    def compute_hinge_loss(self, batched_complex_graphs: dgl.DGLHeteroGraph):

        raise NotImplementedError

        batch_size = len(keypoint_positions)
        n_keypoints = keypoint_positions[0].shape[0]
        n_pairs = n_keypoints*(n_keypoints - 1)/2

        loss = 0
        for kp_pos in keypoint_positions:
            loss += self._hinge_loss(kp_pos)

        loss = loss / batch_size / n_pairs
        return loss