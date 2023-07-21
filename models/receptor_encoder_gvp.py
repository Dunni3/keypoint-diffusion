from typing import Dict, Union

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from einops import rearrange
from torch_cluster import knn, radius, radius_graph
from torch_scatter import segment_csr

from utils import get_batch_info, get_edges_per_batch

from .gvp import GVPEdgeConv

class KeypointInitializer(nn.Module):

    """Assigns initial positions and features to keypoint nodes. Also draws receptor -> keypoint edges."""

    def __init__(self, n_keypoints: int, scalar_size: int, vector_size: int):
        super().__init__()

        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_keypoints = n_keypoints
        self.num_heads = 1

        self.src_net = nn.Linear(scalar_size, scalar_size*self.num_heads, bias=False)
        self.dst_net = nn.Linear(scalar_size, scalar_size*self.num_heads, bias=False)

        # embedding function for the mean node feature before keypoint position generation
        self.keypoint_embedding = nn.Sequential(
            nn.Linear(scalar_size, scalar_size*n_keypoints),
            nn.SiLU(),
            nn.LayerNorm(scalar_size*n_keypoints)
        )

        self.norm = nn.LayerNorm(scalar_size)


    def forward(self, g: dgl.DGLHeteroGraph, rec_scalar_feats: torch.Tensor, batch_idxs: Dict[str, torch.Tensor]):

        # get the device and batch size from the graph, g
        device = g.device
        batch_size = g.batch_size

        with g.local_scope():
            
            g.nodes['rec'].data['h'] = rec_scalar_feats

            # get mean node feature for each graph in the batch
            mean_node_feats = dgl.readout_nodes(g, op='mean', ntype='rec', feat='h')

            # transform mean node features into keypoint embeddings
            keypoint_scalars = self.keypoint_embedding(mean_node_feats)
            keypoint_scalars = rearrange(keypoint_scalars, 'b (k d) -> (b k) d', d=self.scalar_size, k=self.n_keypoints)

            # set keypoint features in the graph
            g.nodes['kp'].data['h'] = keypoint_scalars

            # get queries/keys
            ft_src = self.src_net(rec_scalar_feats).view(-1, self.num_heads, self.scalar_size) 
            ft_dst = self.dst_net(keypoint_scalars).view(-1, self.num_heads, self.scalar_size)

            # Assign features to nodes
            g.srcdata['ft'] = {'rec': ft_src}
            g.dstdata['ft'] = {'kp': ft_dst}

            # Step 1. dot product
            g.apply_edges(fn.u_dot_v('ft', 'ft', 'a'), etype='rk')

            # Step 2. edge softmax to compute attention scores
            a = g.edges['rk'].data['a'] / self.scalar_size**0.5
            a = torch.exp(a)

            edges_per_kp = g.batch_num_nodes('rec')[batch_idxs['kp']] # (n_keypoints*batch_size) -> number of edges going to every keypoint
            indptr = torch.zeros(edges_per_kp.shape[0]+1, dtype=int, device=g.device) # index pointer used for segmented sum
            indptr[1:] = edges_per_kp
            indptr = torch.cumsum(indptr, 0)
            dst_node_denominators = segment_csr(src=a, indptr=indptr) # (n_kp_nodes, num_heads, 1), sum of incoming weights for each node
            g.dstdata['denom'] = {'kp': 1/dst_node_denominators}
            g.apply_edges(fn.v_mul_e('denom', 'a', 'sa'), etype='rk')

            # Step 3. Broadcast softmax value to each edge, and aggregate dst node
            g.update_all(fn.u_mul_e('x_0', 'sa', 'attn'), fn.sum('attn', 'agg_u'), etype='rk')

            # get keypoint positions
            kp_pos = g.dstdata['agg_u']['kp'].mean(dim=1)

        # for now intialize keypoint scalar and vector features to zero
        kp_scalars = torch.zeros(g.num_nodes('kp'), self.scalar_size, device=device, dtype=torch.float32)
        kp_vecs = torch.zeros(g.num_nodes('kp'), self.vector_size, 3, device=device, dtype=torch.float32)

        return kp_pos, kp_scalars, kp_vecs
        
    

class ReceptorEncoderGVP(nn.Module):

    def __init__(self, 
                 in_scalar_size: int, 
                 out_scalar_size: int = 128, 
                 n_message_gvps: int = 1,
                 n_update_gvps: int = 1,
                 vector_size: int = 16,
                 n_rr_convs: int = 3,
                 n_rk_convs: int = 2, 
                 message_norm: Union[float, str] = 10, 
                 use_sameres_feat: bool = False,
                 kp_rad: float = 0, 
                 k_closest: int = 0,
                 dropout: float = 0.0,
                 n_keypoints: int = 20,
                 no_cg: bool = False,
                 graph_cutoffs: dict = {}):
        super().__init__()

        if no_cg:
            raise NotImplementedError('no_cg is not implemented yet')

        if kp_rad != 0 and k_closest != 0:
            raise ValueError('one of kp_rad and kp_closest can be zero but not both')
        elif kp_rad == 0 and k_closest == 0:
            raise ValueError('one of kp_rad and kp_closest must be non-zero')

        self.n_rr_convs = n_rr_convs
        self.n_rk_convs = n_rk_convs
        self.in_scalar_size = in_scalar_size
        self.out_scalar_size = out_scalar_size
        self.n_keypoints = n_keypoints
        self.vector_size = vector_size
        self.dropout_rate = dropout
        self.use_sameres_feat = use_sameres_feat
        self.kp_rad = kp_rad
        self.k_closest = k_closest
        self.message_norm = message_norm
        self.graph_cutoffs = graph_cutoffs

        # check the message norm argument
        if isinstance(message_norm, str) and message_norm != 'mean':
            raise ValueError(f'message norm must be either a float, int, or "mean". Got {message_norm}')
        elif isinstance(message_norm, float) or isinstance(message_norm, int):
            pass
        elif not isinstance(message_norm, (str, float, int)):
            raise ValueError(f'message norm must be either a float, int, or "mean". Got {message_norm}')
        
        # check that either k_closest or kp_rad is non-zero
        if self.k_closest > 0 and self.kp_rad > 0:
            raise ValueError('k_closest and kp_rad cannot both be non-zero')
        elif self.k_closest == 0 and self.kp_rad == 0:
            raise ValueError('k_closest and kp_rad cannot both be zero')

        if self.k_closest > 0:
            self.rk_graph_type = 'knn'
        elif self.kp_rad > 0:
            self.rk_graph_type = 'radius'

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
            self.rr_conv_layers.append(GVPEdgeConv(
                edge_type=('rec', 'rr', 'rec'),
                scalar_size=out_scalar_size,
                vector_size=vector_size,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                edge_feat_size=edge_feat_size,
                dropout=dropout,
                message_norm=message_norm,
                rbf_dmax=graph_cutoffs['rr']
            ))

        # create the keypoint initializer which will assign initial positions to the keypoint nodes
        self.keypoint_initializer = KeypointInitializer(n_keypoints=n_keypoints, scalar_size=out_scalar_size, vector_size=vector_size)

        # create rec-keypoint convolutional layers
        self.rk_conv_layers = nn.ModuleList()
        for i in range(n_rk_convs):

            if i != 0:
                use_dst_feats = True
            else:
                use_dst_feats = False

            self.rk_conv_layers.append(GVPEdgeConv(
                edge_type=('rec', 'rk', 'kp'),
                use_dst_feats=use_dst_feats,
                scalar_size=out_scalar_size,
                vector_size=vector_size,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                edge_feat_size=edge_feat_size,
                dropout=dropout,
                message_norm=message_norm,
                rbf_dmax=graph_cutoffs['rk']
            ))

    def forward(self, g: dgl.DGLHeteroGraph, batch_idxs: Dict[str, torch.Tensor]):

        device = g.device
        batch_size = g.batch_size

        # get scalar features
        rec_scalar_feat = g.nodes['rec'].data["h_0"]

        # embed scalar features
        rec_scalar_feat = self.scalar_embed(rec_scalar_feat)
        rec_scalar_feat = self.scalar_norm(rec_scalar_feat)

        # initialize receptor vector features
        rec_vec_feat = torch.zeros((g.num_nodes('rec'), self.vector_size, 3), device=device)

        # get edge features
        if self.use_sameres_feat:
            edge_feat = g.edges['rr'].data["a"]
        else:
            edge_feat = None

        # get coordinate features
        rec_coord_feat = g.nodes['rec'].data['x_0']

        # compute rec_batch_idx, the batch index of every receptor atom
        rec_batch_idx = batch_idxs['rec']

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
            src_feats = (rec_scalar_feat, rec_coord_feat, rec_vec_feat)
            rec_scalar_feat, rec_vec_feat = self.rr_conv_layers[i](g, src_feats=src_feats, edge_feats=edge_feat, z=z)

        # get initial keypoint positions
        kp_pos, kp_scalars, kp_vecs = self.keypoint_initializer(g, rec_scalar_feat, batch_idxs)

        # set keypoint positions in the graph
        g.nodes['kp'].data['x_0'] = kp_pos

        # update receptor-keypoint edges
        g = self.update_rk_edges(g, batch_idxs)

        # update scalar and vector features of the keypoint nodes
        for i in range(self.n_rk_convs):
            src_feats = (rec_scalar_feat, rec_coord_feat, rec_vec_feat)
            dst_feats = (kp_scalars, kp_pos, kp_vecs)
            kp_scalars, kp_vecs = self.rk_conv_layers[i](g, src_feats=src_feats, dst_feats=dst_feats, z=z)

        # set keypoint scalars and vectors in the graph
        g.nodes['kp'].data['h_0'] = kp_scalars
        g.nodes['kp'].data['v_0'] = kp_vecs

        # get batch info 
        batch_num_nodes, batch_num_edges = get_batch_info(g)

        # add keypoint-keypoint edges
        kk_edges = radius_graph(x=kp_pos, r=self.graph_cutoffs['kk'], batch=batch_idxs['kp'], max_num_neighbors=100)
        g.add_edges(kk_edges[0], kk_edges[1], etype='kk')

        # get number of keypoint-keypoint edges in each batch
        batch_num_edges[('kp', 'kk', 'kp')] = get_edges_per_batch(kk_edges[0], batch_size, batch_idxs['kp'])

        g.set_batch_num_nodes(batch_num_nodes)
        g.set_batch_num_edges(batch_num_edges)

        return g


    def update_rk_edges(self, g: dgl.DGLHeteroGraph, batch_idxs: Dict[str, torch.Tensor]):

        kp_pos = g.nodes['kp'].data['x_0']
        rec_pos = g.nodes['rec'].data['x_0']

        if self.rk_graph_type == 'knn':
            # find edges of KNN graph where each keypoint node has incoming edges from the K closest receptor nodes
            rk_idxs = knn(x=rec_pos, y=kp_pos, k=self.k_closest, batch_x=batch_idxs['rec'], batch_y=batch_idxs['kp'])
        elif self.rk_graph_type == 'radius':
            rk_idxs = radius(x=rec_pos, y=kp_pos, r=self.kp_rad, batch_x=batch_idxs['rec'], batch_y=batch_idxs['kp'], max_num_neighbors=10)

        # we are going to remove and then add edges which destroy batch information. 
        # we have to record batch info before mutating graph topology and add it back afterwards
        batch_num_nodes, batch_num_edges = get_batch_info(g)

        # get number of receptor-keypoint edges for each graph in the batch
        if self.rk_graph_type == 'knn':
            batch_num_edges[('rec', 'rk', 'kp')] = torch.ones(g.batch_size, device=g.device, dtype=int)*self.n_keypoints*self.k_closest
        elif self.rk_graph_type == 'radius':
            batch_num_edges[('rec', 'rk', 'kp')] = get_edges_per_batch(rk_idxs[0], g.batch_size, batch_idxs['kp'])

        g.remove_edges(g.edges(form='eid', etype='rk'), etype='rk') # remove all receptor-keypoint edges
        g.add_edges(rk_idxs[1], rk_idxs[0], etype='rk') # add edges that we just computed
    
        # reset batch info
        g.set_batch_num_nodes(batch_num_nodes)
        g.set_batch_num_edges(batch_num_edges)

        return g