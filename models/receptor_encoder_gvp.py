import torch.nn as nn
import torch
import dgl
import dgl.function as fn
from typing import Union

from .gvp import GVP, GVPDropout, GVPLayerNorm

class ReceptorConv(nn.Module):
    # this is adapted from the EGNN implementation in DGL

    def __init__(self, scalar_size: int = 128, vector_size: int = 16,
                  edge_feat_size=0, use_tanh=True, coords_range=10, message_norm: Union[float, str] = 10, dropout: float = 0.0):
        super(ReceptorConv, self).__init__()

        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()
        self.use_tanh = use_tanh
        self.coords_range = coords_range
        self.message_norm = message_norm
        self.dropout_rate = dropout
        
        self.edge_message = GVP(dim_vectors_in=vector_size+1, 
                                dim_vectors_out=vector_size, 
                                dim_feats_in=scalar_size, 
                                dim_feats_out=scalar_size, 
                                feats_activation=act_fn, 
                                vectors_activation=act_fn, 
                                vector_gating=True, 
                                dropout_rate=dropout, 
                                layer_norm=True)
        
        self.node_update = GVP(dim_vectors_in=vector_size, 
                               dim_vectors_out=vector_size+1, 
                               dim_feats_in=scalar_size, 
                               dim_feats_out=scalar_size, 
                               feats_activation=act_fn, 
                               vectors_activation=act_fn, 
                               vector_gating=True, 
                               dropout_rate=dropout, 
                               layer_norm=True)
        
        self.dropout = GVPDropout(self.dropout_rate)
        self.layer_norm = GVPLayerNorm(self.scalar_size)

        if self.message_norm == 'mean':
            self.agg_func = fn.mean
        else:
            self.agg_func = fn.sum

    def forward(self, g: dgl.DGLHeteroGraph, scalar_feat: torch.Tensor, coord_feat: torch.Tensor, vec_feat: torch.Tensor, z: Union[float, torch.Tensor] = 1, edge_feat: torch.Tensor=None):

        # vec_feat has shape (n_nodes, n_vectors, 3)

        with g.local_scope():
            g.nodes['rec'].data["h"] = scalar_feat
            g.nodes['rec'].data["x"] = coord_feat
            g.nodes['rec'].data["v"] = vec_feat

            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                g.edges['rr'].data["a"] = edge_feat

            # get vectors between receptor nodes
            g.apply_edges(fn.u_sub_v("x", "x", "x_diff"), etype='rr')

            # normalize x_diff
            # i don't think this is necessary but i'm leaving it here for now
            # g.edges['rr'].data['x_diff'] = g.edges['rr'].data['x_diff'] / (torch.norm(g.edges['rr'].data['x_diff'], dim=-1, keepdim=True) + 1e-10 )

            # copy source node features to edges
            g.apply_edges(fn.copy_u("h", "h"), etype='rr')
            g.apply_edges(fn.copy_u("v", "v"), etype='rr')

            # compute messages on every receptor-receptor edge
            g.apply_edges(self.message, etype='rr')

            # aggregate messages from every receptor-receptor edge
            g.update_all(fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg"), etype='rr')
            g.update_all(fn.copy_e("vec_msg", "m"), self.agg_func("m", "vec_msg"), etype='rr')

            # get scalar and vector messages
            scalar_msg = g.nodes['rec'].data["scalar_msg"] / z
            vec_msg = g.nodes['rec'].data["vec_msg"] / z

            # dropout scalar and vector messages
            scalar_msg, vec_msg = self.dropout((scalar_msg, vec_msg))

            # update scalar and vector features, apply layernorm
            scalar_feat = scalar_feat + scalar_msg
            vec_feat = vec_feat + vec_msg
            scalar_feat, vec_feat = self.layer_norm((scalar_feat, vec_feat))

            # apply node update function, apply dropout to residuals, apply layernorm
            scalar_residual, vec_residual = self.node_update((scalar_feat, vec_feat))
            scalar_residual, vec_residual = self.dropout((scalar_residual, vec_residual))
            scalar_feat = scalar_feat + scalar_residual
            vec_feat = vec_feat + vec_residual
            scalar_feat, vec_feat = self.layer_norm((scalar_feat, vec_feat))

        return scalar_feat, vec_feat

    def message(self, edges):

        # concatenate x_diff and v on every edge to produce vector features
        vec_feats = torch.cat([edges.data["x_diff"].unsqueeze(1), edges.data["v"]], dim=1)

        # create scalar features
        if self.edge_feat_size > 0:
            scalar_feats = torch.cat([edges.data["h"], edges.data['a'] ])
        else:
            scalar_feats = edges.data["h"]

        scalar_message, vector_message = self.edge_message((scalar_feats, vec_feats))

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}
    

class ReceptorEncoderGVP(nn.Module):

    def __init__(self, 
                 in_scalar_size: int, 
                 out_scalar_size: int = 128, 
                 vector_size: int = 16,
                 n_keypoints: int = 20,
                 n_rr_convs: int = 3, 
                 message_norm: Union[float, str] = 10, 
                 use_sameres_feat: bool = False,
                 kp_feat_scale=1, 
                 kp_rad: float = 0, 
                 k_closest: int = 0,
                 dropout: float = 0.0,
                 graph_cutoffs: dict = {}):
        super().__init__()

        if kp_rad != 0 and k_closest != 0:
            raise ValueError('one of kp_rad and kp_closest can be zero but not both')
        elif kp_rad == 0 and k_closest == 0:
            raise ValueError('one of kp_rad and kp_closest must be non-zero')

        self.n_rr_convs = n_rr_convs
        self.in_scalar_size = in_scalar_size
        self.out_scalar_size = out_scalar_size
        self.n_keypoints = n_keypoints
        self.vector_size = vector_size
        self.dropout_rate = dropout
        self.use_sameres_feat = use_sameres_feat
        self.kp_feat_scale = kp_feat_scale
        self.kp_rad = kp_rad
        self.k_closest = k_closest
        self.message_norm = message_norm
        self.graph_cutoffs = graph_cutoffs

        # check the message norm argument
        if isinstance(message_norm, str):
            if message_norm != 'mean':
                raise ValueError('message norm must be either a float or "mean"')
        elif isinstance(message_norm, float) or isinstance(message_norm, int):
            pass
        else:
            raise ValueError('message norm must be either a float, int, or "mean"')

        # create functions to embed scalar features to the desired size
        self.scalar_embed = nn.Sequential(
            nn.Linear(in_scalar_size, out_scalar_size),
            nn.SiLU(),
            nn.Linear(out_scalar_size, out_scalar_size),
            nn.SiLU()
        )
        self.scalar_norm = nn.LayerNorm(out_scalar_size)

        # set the edge feature size for rec-rec convolutions
        if self.use_sameres_feat:
            edge_feat_size = 1
        else:
            edge_feat_size = 0

        # create rec-rec convolutional layers
        self.rr_conv_layers = nn.ModuleList()
        for _ in range(n_rr_convs):
            self.rr_conv_layers.append(ReceptorConv(
                scalar_size=out_scalar_size,
                vector_size=vector_size,
                edge_feat_size=edge_feat_size,
                dropout=dropout,
                message_norm=message_norm
            ))

    def forward(self, g: dgl.DGLHeteroGraph, kp_batch_idx: torch.Tensor):

        device = g.device
        batch_size = g.batch_size

        with g.local_scope():

            # get scalar features
            scalar_feat = g.nodes['rec'].data["h"]

            # embed scalar features
            scalar_feat = self.scalar_embed(scalar_feat)
            scalar_feat = self.scalar_norm(scalar_feat)

            # initialize receptor vector features
            vec_feat = torch.zeros((g.num_nodes('rec'), self.vector_size, 3), device=device)

            # get edge features
            if self.use_sameres_feat:
                edge_feat = g.edges['rr'].data["a"]
            else:
                edge_feat = None

            # get coordinate features
            coord_feat = g.nodes['rec'].data['x_0']

            # compute rec_batch_idx, the batch index of every receptor atom
            rec_batch_idx = torch.arange(batch_size, device=device).repeat_interleave(g.batch_num_nodes('rec'))

            # compute the normalization factor for the messages if necessary
            if self.message_norm == 'mean':
                # set z to 1. the receptor convolutional layer will use mean aggregation instead of sum
                z = 1
            elif self.message_norm == 0:
                # if messsage_norm is 0, we normalize by the average in-degree of the graph
                z = g.batch_num_edges(etype='rr') / g.batch_num_nodes('rec')
                z = z[rec_batch_idx].view(-1, 1)
            else:
                # otherwise, message_norm is a non-zero constant which we use as the normalization factor
                z = self.message_norm

            # apply receptor-receptor convolutions
            for i in range(self.n_rr_convs):
                scalar_feat, vec_feat = self.rr_conv_layers[i](g, scalar_feat, coord_feat, vec_feat, z=z, edge_feat=edge_feat)

        raise NotImplementedError('ReceptorEncoderGVP is not yet implemented')


