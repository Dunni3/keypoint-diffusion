import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from typing import Dict

from utils import get_batch_info

class FixedReceptorEncoder(nn.Module):

    def __init__(self, n_vec_feats: int, graph_cutoffs: Dict[str, float]):
        
        self.n_vec_feats = n_vec_feats
        self.graph_cutoffs = graph_cutoffs

    def forward(self, g: dgl.DGLHeteroGraph, batch_idxs: Dict[str, torch.Tensor]):
        
        device = g.device
        batch_size = g.batch_size

        # get initial batch information
        batch_num_nodes, batch_num_edges = get_batch_info(g)

        # remove keypoint nodes
        g = g.remove_nodes(g.nodes('kp'), ntype='kp')

        # add the same number of keypoint nodes as there are receptor nodes
        g.add_nodes(
            g.num_nodes('rec'),
            {'x_0': g.nodes['rec'].data['x_0'], 'h_0':g.nodes['rec'].data['h_0']},
            ntype='kp'
        )

        # add vector features to keypoint nodes
        if self.n_vec_feats is not None:
            g.nodes['kp'].data['v_0'] = torch.zeros((g.num_nodes('kp'), self.n_vec_feats, 3), device=device)

        # copy rec-rec edges to kp-kp edges
        g.add_edges(
            *g.edges('rr'),
            etype='kk'
        )

        batch_num_nodes['kp'] = batch_num_nodes['rec']
        batch_num_edges[('kp', 'kk', 'kp')] = batch_num_edges[('rec', 'rr', 'rec')]

        g.set_batch_num_nodes(batch_num_nodes)
        g.set_batch_num_edges(batch_num_edges)

        # remove rec nodes
        g.remove_nodes(g.nodes('rec'), ntype='rec')

        assert batch_size == g.batch_size # check that batch information was preserved

        return g


