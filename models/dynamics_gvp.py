import torch.nn as nn
import dgl
import torch
from typing import Dict, List, Tuple, Union

from utils import get_batch_info, get_edges_per_batch
from torch_cluster import radius_graph, knn_graph, knn, radius
from .gvp import GVPMultiEdgeConv, GVP

class NoisePredictionBlock(nn.Module):

    def __init__(self, in_scalar_dim: int, out_scalar_dim: int, vector_size: int, n_gvps: int = 3, intermediate_scalar_dim: int = 64):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):

            if i == n_gvps - 1:
                dim_vectors_out = 1
                dim_feats_out = intermediate_scalar_dim
                vectors_activation = nn.Identity()
            else:
                dim_vectors_out = vector_size
                dim_feats_out = in_scalar_dim
                vectors_activation = nn.Sigmoid()

            self.gvps.append(GVP(
                dim_vectors_in=vector_size,
                dim_vectors_out=dim_vectors_out,
                dim_feats_in=in_scalar_dim,
                dim_feats_out=dim_feats_out,
                vectors_activation=vectors_activation
            ))
        self.gvps = nn.Sequential(*self.gvps)

        self.to_scalar_output = nn.Linear(intermediate_scalar_dim, out_scalar_dim)

    def forward(self, lig_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):

        scalars, _, vectors = lig_data
        scalars, vectors = self.gvps((scalars, vectors))
        scalars = self.to_scalar_output(scalars)
        return scalars, vectors

class LigRecGVP(nn.Module):

    no_kp_update_edges = [
            ('lig', 'll', 'lig'),
            ('kp', 'kl', 'lig'),
        ]
    
    kp_update_edges = no_kp_update_edges + [ 
        ('lig', 'lk', 'kp'),
        ('kp', 'kk', 'kp'), 
        ]

    def __init__(self, in_scalar_dim: int, in_vector_dim: int, out_scalar_dim: int, update_kp: bool = False, n_convs: int = 4,
                 n_message_gvps: int = 3, n_update_gvps: int = 2, message_norm: Union[float, str, Dict] = 10, n_noise_gvps: int = 3, dropout: float = 0.0):
        super().__init__()

        self.update_kp = update_kp

        self.conv_layers = nn.ModuleList()
        for i in range(n_convs):

            if not update_kp:
                edge_types = self.no_kp_update_edges
            elif update_kp and i != n_convs - 1:
                edge_types = self.kp_update_edges
            elif update_kp and i == n_convs - 1:
                edge_types = self.no_kp_update_edges
            else:
                raise ValueError('not sure how you get here but ok')

            self.conv_layers.append(GVPMultiEdgeConv(
                etypes=edge_types,
                scalar_size=in_scalar_dim,
                vector_size=in_vector_dim,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                message_norm=message_norm,
                dropout=dropout
            ))

        self.noise_predictor = NoisePredictionBlock(
            in_scalar_dim=in_scalar_dim,
            out_scalar_dim=out_scalar_dim,
            vector_size=in_vector_dim,
            n_gvps=n_noise_gvps
        )

    def forward(self, g: dgl.DGLHeteroGraph, node_data: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):

        # do message passing between ligand atoms and keypoints
        for conv_layer in self.conv_layers:
            node_data = conv_layer(g, node_data)

        # predict noise on ligand atoms
        scalar_noise, vector_noise = self.noise_predictor(node_data['lig'])
        return scalar_noise, vector_noise


class LigRecDynamicsGVP(nn.Module):

    def __init__(self, n_lig_scalars, n_kp_scalars, vector_dim: int = 16, n_convs=4, n_hidden_scalars=128, act_fn=nn.SiLU,
                 message_norm=1, no_cg: bool = False, n_keypoints: int = 20, graph_cutoffs: dict = {}, update_kp: bool = False, 
                 ll_k: int = 0, kl_k: int = 0, n_message_gvps: int = 3, n_update_gvps: int = 2, n_noise_gvps: int = 3, dropout: float = 0.0):
        super().__init__()

        if no_cg:
            raise NotImplementedError("No CG is not implemented for GVP")
        

        self.n_keypoints = n_keypoints
        self.graph_cutoffs = graph_cutoffs
        self.update_kp = update_kp
        self.n_lig_scalars = n_lig_scalars
        self.vector_dim = vector_dim

        self.ll_k = ll_k
        self.kl_k = kl_k

        self.lig_encoder = nn.Sequential(
            nn.Linear(n_lig_scalars+1, n_hidden_scalars),
            act_fn(),
            nn.LayerNorm(n_hidden_scalars)
        )

        self.kp_encoder = nn.Sequential(
            nn.Linear(n_kp_scalars+1, n_hidden_scalars),
            act_fn(),
            nn.LayerNorm(n_hidden_scalars)
        )

        self.noise_predictor = LigRecGVP(
            in_scalar_dim=n_hidden_scalars,
            in_vector_dim=vector_dim,
            out_scalar_dim=n_lig_scalars,
            update_kp=update_kp,
            n_convs=n_convs,
            n_message_gvps=n_message_gvps,
            n_update_gvps=n_update_gvps,
            n_noise_gvps=n_noise_gvps,
            message_norm=message_norm,
            dropout=dropout
        )

    def forward(self, g: dgl.DGLHeteroGraph, timestep: torch.Tensor, batch_idxs: Dict[str, torch.Tensor]):

        lig_batch_idx = batch_idxs['lig']
        kp_batch_idx = batch_idxs['kp']

        with g.local_scope():

            # get initial lig and rec features from graph
            lig_scalars = g.nodes['lig'].data['h_0']
            kp_scalars = g.nodes['kp'].data['h_0']

            # add timestep to node features
            t_lig = timestep[lig_batch_idx].view(-1, 1)
            lig_scalars = torch.concatenate([lig_scalars, t_lig], dim=1)

            t_kp = timestep[kp_batch_idx].view(-1, 1)
            kp_scalars = torch.concatenate([kp_scalars, t_kp], dim=1)

            # encode lig/kp scalars into a space of the same dimensionality
            lig_scalars = self.lig_encoder(lig_scalars)
            kp_scalars = self.kp_encoder(kp_scalars)

            # set lig/kp features in graph
            g.nodes['lig'].data['h_0'] = lig_scalars
            g.nodes['kp'].data['h_0'] = kp_scalars
            g.nodes['lig'].data['v_0'] = torch.zeros((lig_scalars.shape[0], self.vector_dim, 3),
                                                     device=g.device, dtype=lig_scalars.dtype)
            
            # construct node data for noise predictor
            node_data = {}
            node_data['lig'] = (
                lig_scalars,
                g.nodes['lig'].data['x_0'],
                torch.zeros((lig_scalars.shape[0], self.vector_dim, 3),
                            device=g.device, dtype=lig_scalars.dtype)
            )
            node_data['kp'] = (
                kp_scalars,
                g.nodes['kp'].data['x_0'],
                g.nodes['kp'].data['v_0']
            )

            # add lig-lig and kp<->lig edges to graph
            g = self.add_lig_edges(g, lig_batch_idx, kp_batch_idx)

            # predict noise
            eps_h, eps_x = self.noise_predictor(g, node_data)

            self.remove_lig_edges(g)

        return eps_h, eps_x

    def add_lig_edges(self, g: dgl.DGLHeteroGraph, lig_batch_idx, kp_batch_idx) -> dgl.DGLHeteroGraph:

        batch_num_nodes, batch_num_edges = get_batch_info(g)
        batch_size = g.batch_size

        # add lig-lig edges
        if self.ll_k > 0: # if ll_k is 0, we use radius graph, otherwise we use knn graphs with ll_k neighbors
            ll_idxs = knn_graph(g.nodes['lig'].data['x_0'], k=self.ll_k, batch=lig_batch_idx)
        else:
            ll_idxs = radius_graph(g.nodes['lig'].data['x_0'], r=self.graph_cutoffs['ll'], batch=lig_batch_idx, max_num_neighbors=200)
        g.add_edges(ll_idxs[0], ll_idxs[1], etype='ll')

        # compute kp -> lig edges
        if self.kl_k > 0: # if kl_k is 0, we use radius graph, otherwise we use knn graphs with kl_k neighbors
            kl_idxs = knn(x=g.nodes['lig'].data['x_0'], y=g.nodes['kp'].data['x_0'], k=self.kl_k, batch_x=lig_batch_idx, batch_y=kp_batch_idx)
        else:
            kl_idxs = radius(x=g.nodes['lig'].data['x_0'], y=g.nodes['kp'].data['x_0'], batch_x=lig_batch_idx, batch_y=kp_batch_idx, r=self.graph_cutoffs['kl'], max_num_neighbors=100)
        g.add_edges(kl_idxs[0], kl_idxs[1], etype='kl')

        # compute batch information
        batch_num_edges[('lig', 'll', 'lig')] = get_edges_per_batch(ll_idxs[0], batch_size, lig_batch_idx)
        kl_edges_per_batch = get_edges_per_batch(kl_idxs[0], batch_size, kp_batch_idx)
        batch_num_edges[('kp', 'kl', 'lig')] = kl_edges_per_batch

        # add lig -> kp edges if necessary
        if self.update_kp:
            g.add_edges(kl_idxs[1], kl_idxs[0], etype='lk')
            batch_num_edges[('lig', 'lk', 'kp')] = kl_edges_per_batch

        # update the graph's batch information
        g.set_batch_num_edges(batch_num_edges)
        g.set_batch_num_nodes(batch_num_nodes)

        return g
    
    def remove_lig_edges(self, g: dgl.DGLHeteroGraph):

        if self.update_kp:
            etypes_to_remove = ['ll', 'kl', 'lk']
        else:
            etypes_to_remove = ['ll', 'kl']
        
        batch_num_nodes, batch_num_edges = get_batch_info(g)

        for canonical_etype in batch_num_edges:
            if canonical_etype[1] in etypes_to_remove:
                batch_num_edges[canonical_etype] = torch.zeros_like(batch_num_edges[canonical_etype])
        
        for etype in etypes_to_remove:
            eids = g.edges(form='eid', etype=etype)
            g.remove_edges(eids, etype=etype)
        
        g.set_batch_num_nodes(batch_num_nodes)
        g.set_batch_num_edges(batch_num_edges)

        return g