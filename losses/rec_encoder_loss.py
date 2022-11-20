from torch.nn.modules.loss import _Loss
import torch
import dgl
import ot
from typing import List

# this function is taken from equibind
def compute_ot_emd(cost_mat, device):
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    a = np.ones([cost_mat.shape[0]]) / cost_mat.shape[0]
    b = np.ones([cost_mat.shape[1]]) / cost_mat.shape[1]
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached

class ReceptorEncoderLoss(_Loss):

    def __init__(self, use_boltzmann_weights=False):
        self.use_boltzmann_weights = use_boltzmann_weights

    def forward(self, keypoint_positions: List[torch.Tensor], rec_graphs: dgl.DGLGraph, boltzmann_weights=None):
        # TODO: incorporate boltzmann-weights into cost matrix
        # compute cost matrix
        ot_loss = 0

        unbatched_graphs = dgl.unbatch(rec_graphs)

        for kp_pos, rec_graph in zip(keypoint_positions, unbatched_graphs):
            # compute cost matrix
            cost_mat = torch.square(torch.cdist(kp_pos, rec_graph.ndata['x_0']))
            # compute OT distance
            # TODO: what should device be? not really sure how devices work in pytorch...need to figure that out
            ot_dist, _ = compute_ot_emd(cost_mat, device=cost_mat.device)
            ot_loss += ot_dist
        
        ot_loss = ot_loss / len(unbatched_graphs)

        return ot_loss