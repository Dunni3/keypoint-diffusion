from typing import Dict, List

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax
from einops import rearrange
from torch_cluster import radius_graph, radius, knn
from torch_scatter import segment_csr

from utils import get_batch_info, get_edges_per_batch

class ReceptorConv(nn.Module):
    # this is adapted from the EGNN implementation in DGL

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, use_tanh=True, coords_range=10, message_norm=1, fix_pos: bool = False, norm: bool = False):
        super(ReceptorConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()
        self.use_tanh = use_tanh
        self.coords_range = coords_range
        self.message_norm = message_norm
        self.fix_pos = fix_pos
        self.norm = norm

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

        self.soft_attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        if self.norm:
            self.layer_norm = nn.LayerNorm(out_size)
        else:
            self.layer_norm = nn.Identity()

        if self.fix_pos:
            return

        # \phi_x
        coord_output_layer = nn.Linear(hidden_size, 1, bias=False)
        nn.init.xavier_uniform_(coord_output_layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            coord_output_layer
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["radial"],
                    edges.data["a"],
                ],
                dim=-1,
            )
        else:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["radial"]], dim=-1
            )

        msg_h = self.edge_mlp(f)
        msg_h = msg_h*self.soft_attention(msg_h)
        if self.fix_pos:
            msg_x = torch.zeros_like(edges.data["radial"])
        elif self.use_tanh:
            msg_x = torch.tanh( self.coord_mlp(f) ) * edges.data["x_diff"] * self.coords_range
        else:
            msg_x = self.coord_mlp(f) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, g: dgl.DGLHeteroGraph, node_feat: torch.Tensor, coord_feat: torch.Tensor, z: torch.Tensor, edge_feat: torch.Tensor=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        coord_feat : torch.Tensor
            The coordinate feature of shape :math:`(N, h_x)`. :math:`N` is the
            number of nodes, and :math:`h_x` can be any positive integer.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        node_feat_out : torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            is the same as out_size.
        coord_feat_out: torch.Tensor
            The output coordinate feature of shape :math:`(N, h_x)` where :math:`h_x`
            is the same as the input coordinate feature dimension.
        """
        with g.local_scope():
            # node feature
            g.nodes['rec'].data["h"] = node_feat
            # coordinate feature
            g.nodes['rec'].data["x"] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                g.edges['rr'].data["a"] = edge_feat
            # get coordinate diff & radial features
            g.apply_edges(fn.u_sub_v("x", "x", "x_diff"), etype='rr')
            g.edges['rr'].data["radial"] = torch.norm(g.edges['rr'].data["x_diff"], dim=1).unsqueeze(-1)
            # normalize coordinate difference
            g.edges['rr'].data["x_diff"] = g.edges['rr'].data["x_diff"] / (
                g.edges['rr'].data["radial"] + 1
            )
            g.apply_edges(self.message, etype='rr')
            g.update_all(fn.copy_e("msg_x", "m"), fn.sum("m", "x_neigh"), etype='rr')
            g.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"), etype='rr')

            h_neigh, x_neigh = g.nodes['rec'].data["h_neigh"]/z, g.nodes['rec'].data["x_neigh"]/z

            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh

            h = self.layer_norm(h)

            return h, x
        
class RecKeyConv(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, n_keypoints: int, num_heads: int = 1, k_closest: int = 0, kp_rad: float = 0, fix_pos: bool = False, norm: bool = False):
        super().__init__()

        self.num_heads = num_heads
        self.out_feats = out_feats
        self.n_keypoints = n_keypoints
        self.fix_pos = fix_pos
        self.k_closest = k_closest
        self.kp_rad = kp_rad
        self.norm = norm

        self.fc_src = nn.Linear(in_feats, out_feats*num_heads, bias=False)
        self.fc_dst = nn.Linear(in_feats, out_feats*num_heads, bias=False)

        self.kp_feature_mlp = nn.Sequential(
            nn.Linear(out_feats+self.k_closest, out_feats),
            nn.SiLU()
        )

        if self.norm:
            self.layer_norm = nn.LayerNorm(out_feats)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, g: dgl.DGLHeteroGraph, kp_batch_idx):

        with g.local_scope():

            h_src = g.nodes['rec'].data['h']
            h_dst = g.nodes['kp'].data['h_0']

            # get queries/keys
            ft_src = self.fc_src(h_src).view(-1, self.num_heads, self.out_feats) 
            ft_dst = self.fc_src(h_dst).view(-1, self.num_heads, self.out_feats)

            # Assign features to nodes
            g.srcdata['ft'] = {'rec': ft_src}
            g.dstdata['ft'] = {'kp': ft_dst}

            # Step 1. dot product
            g.apply_edges(fn.u_dot_v('ft', 'ft', 'a'), etype='rk')

            # Step 2. edge softmax to compute attention scores
            a = g.edges['rk'].data['a'] / self.out_feats**0.5
            a = torch.exp(a)

            edges_per_kp = g.batch_num_nodes('rec')[kp_batch_idx] # (n_keypoints*batch_size) -> number of edges going to every keypoint
            indptr = torch.zeros(edges_per_kp.shape[0]+1, dtype=int, device=g.device) # index pointer used for segmented sum
            indptr[1:] = edges_per_kp
            indptr = torch.cumsum(indptr, 0)
            dst_node_denominators = segment_csr(src=a, indptr=indptr) # (n_kp_nodes, num_heads, 1), sum of incoming weights for each node
            g.dstdata['denom'] = {'kp': 1/dst_node_denominators}
            g.apply_edges(fn.v_mul_e('denom', 'a', 'sa'), etype='rk')

            # Step 3. Broadcast softmax value to each edge, and aggregate dst node
            if self.fix_pos:
                val_str = 'x_0'
            else:
                val_str = 'x'
            g.update_all(fn.u_mul_e(val_str, 'sa', 'attn'), fn.sum('attn', 'agg_u'), etype='rk')

            # get keypoint positions
            kp_pos = g.dstdata['agg_u']['kp'].mean(dim=1)
            g.nodes['kp'].data['x'] = kp_pos


            # get keypoint features
            if self.k_closest != 0:
                kp_feat = self.k_closest_feats(g)
            elif self.kp_rad != 0:
                kp_feat = self.kp_rad_feats(g, kp_batch_idx=kp_batch_idx)
            else:
                raise NotImplementedError

            kp_feat = self.kp_feature_mlp(kp_feat)
            kp_feat = self.layer_norm(kp_feat)


        return kp_pos, kp_feat
    
    def kp_rad_feats(self, g: dgl.DGLHeteroGraph, kp_batch_idx: torch.Tensor):

        # we are going to remove and then add edges which destroy batch information. we have to record batch info before mutating
        # graph topology and add it back afterwards
        batch_num_nodes, batch_num_edges = get_batch_info(g)

        kp_pos = g.nodes['kp'].data['x']
        batch_idxs = torch.arange(g.batch_size, device=g.device)
        rec_atom_batch = batch_idxs.repeat_interleave(g.batch_num_nodes('rec'))
        rad_idxs = radius(x=g.nodes['rec'].data['x_0'], y=kp_pos, batch_x=rec_atom_batch, batch_y=kp_batch_idx, r=self.kp_rad, max_num_neighbors=100) # shape (2, n_keypoints*?*batch_size)

        # get number of edges corresponding to each batch
        batch_num_edges[('rec', 'rk', 'kp')] = get_edges_per_batch(rad_idxs[0], g.batch_size, kp_batch_idx)

        g.remove_edges(g.edges(form='eid', etype='rk'), etype='rk') # remove all receptor-keypoint edges
        g.add_edges(rad_idxs[1], rad_idxs[0], etype='rk') # add back the edges identified by radius

        # reset batch info
        g.set_batch_num_nodes(batch_num_nodes)
        g.set_batch_num_edges(batch_num_edges)

        # accumulate features from receptors in each keypoint's neighborhood
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_m'), etype='rk')
        z = g.batch_num_edges('rk') / g.batch_num_nodes('kp')
        z = z[kp_batch_idx].view(-1, 1) + 1 
        kp_feats = g.nodes['kp'].data['h_m']/z
        return kp_feats

    def k_closest_feats(self, g: dgl.DGLHeteroGraph):
        # get k edges having the lowest distance to each keypoint
        kp_pos = g.nodes['kp'].data['x']

        batch_idxs = torch.arange(g.batch_size, device=g.device)
        rec_atom_batch = batch_idxs.repeat_interleave(g.batch_num_nodes('rec'))
        kp_batch = batch_idxs.repeat_interleave(g.batch_num_nodes('kp'))
        knn_idxs = knn(x=g.nodes['rec'].data['x_0'], y=kp_pos, batch_x=rec_atom_batch, batch_y=kp_batch, k=self.k_closest) # shape (2, n_keypoints*k*batch_size)


        # we are going to remove and then add edges which destroy batch information. we have to record batch info before mutating
        # graph topology and add it back afterwards
        batch_num_nodes, batch_num_edges = get_batch_info(g)

        # we have to change the number of rk edges per batch because we will be altering that
        batch_num_edges['rk'] = torch.ones(g.batch_size, device=g.device, dtype=int)*self.n_keypoints*self.k_closest

        g.remove_edges(g.edges(form='eid', etype='rk'), etype='rk') # remove all receptor-keypoint edges
        g.add_edges(knn_idxs[1], knn_idxs[0], etype='rk') # add back the edges identified by knn

        # reset batch info
        g.set_batch_num_nodes(batch_num_nodes)
        g.set_batch_num_edges(batch_num_edges)

        # get mean rec feature on every keypoint
        g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_m'), etype='rk')
        g.apply_edges(fn.u_sub_v('x', 'x', 'x_diff'), etype='rk')
        g.edges['rk'].data['d'] = torch.norm(g.edges['rk'].data['x_diff']+1e-30, dim=1)
        g.update_all(fn.copy_e('d', 'd'), self.collect_dists, etype='rk')
        
        kp_feat = torch.concatenate([g.nodes['kp'].data['h_m'], g.nodes['kp'].data['d_k']], dim=1)
        return kp_feat

    def collect_dists(self, nodes):
        return {'d_k': nodes.mailbox['d']}

    
class KeyKeyConv(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, num_heads: int = 1, pre_norm=False, post_norm=True) -> None:
        super().__init__()

        self.head_size = in_feats // num_heads 
        self.num_heads = num_heads 

        self.fc_src = nn.Linear(in_feats, self.head_size*num_heads, bias=False)
        self.fc_dst = nn.Linear(in_feats, self.head_size*num_heads, bias=False)
        self.val_fn = nn.Linear(in_feats, self.head_size*num_heads, bias=False)

        self.merge_heads = nn.Linear(self.head_size*num_heads, out_feats, bias=False)

        # self.norm = nn.LayerNorm()
        if pre_norm:
            self.pre_norm = nn.LayerNorm(in_feats)
        else:
            self.pre_norm = nn.Identity()

        if post_norm:
            self.post_norm = nn.LayerNorm(out_feats)
        else:
            self.post_norm = nn.Identity()

        self.dense = nn.Sequential(
            nn.Linear(out_feats, out_feats*2),
            nn.SiLU(),
            nn.Linear(out_feats*2, out_feats),
            nn.SiLU()
        )

    def forward(self, g: dgl.DGLHeteroGraph):

        raise NotImplementedError
        # TODO: I really want to add euclidean distance as something used to compute edge features here. Right now we do atteniton on keypoint features but include no information about
        # relative position other than through the connectivity of the graph

        with g.local_scope():

            batch_num_nodes, batch_num_edges = get_batch_info(g)
            
            batch_num_nodes = { k:v for k,v in batch_num_nodes.items() if k == 'kp'}
            batch_num_edges = { k:v for k,v in batch_num_edges.items() if k[1] == 'kk' }

            g_kp = dgl.to_homogeneous(dgl.node_type_subgraph(g, ntypes=['kp']), ndata=['h_0'])

            g_kp.set_batch_num_edges(g.batch_num_edges('kk'))
            g_kp.set_batch_num_nodes(g.batch_num_nodes('kp'))

            h_src = h_dst = self.pre_norm(g_kp.ndata['h_0'])

            # get queries/keys
            ft_src = self.fc_src(h_src).view(-1, self.num_heads, self.head_size) 
            ft_dst = self.fc_src(h_dst).view(-1, self.num_heads, self.head_size)

            # Assign features to nodes
            g_kp.srcdata['ft'] = ft_src
            g_kp.dstdata['ft'] = ft_dst
            g_kp.srcdata['val'] = self.val_fn(g_kp.ndata['h_0']).view(-1, self.num_heads, self.head_size)

            # Step 1. dot product
            g_kp.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

            # Step 2. edge softmax to compute attention scores
            g_kp.edata['a'] = g_kp.edata['a'] / self.head_size**0.5 
            g_kp.edata['sa'] = edge_softmax(g_kp, g_kp.edata['a'], norm_by='dst')

            # multiply values by attention weights and collect sums on destination nodes
            g_kp.update_all(fn.v_mul_e('val', 'sa', 'msg'), fn.sum('msg', 'h_att'))
            h_att = self.merge_heads(rearrange(g_kp.ndata['h_att'], 'n h d -> n (h d)'))

            h = h_src + h_att

        h = self.dense(self.post_norm(h))

        return h

class ReceptorEncoder(nn.Module):

    def __init__(self, n_convs: int = 6, n_keypoints: int = 10, graph_cutoffs: dict = {}, in_n_node_feat: int = 13, 
                use_sameres_feat: bool = False,
                hidden_n_node_feat: int = 256, 
                out_n_node_feat: int = 256, 
                use_tanh=True, 
                coords_range=10, 
                kp_feat_scale=1, 
                message_norm=1,
                kp_rad: float = 0, 
                k_closest: int = 0,
                norm: bool = False, 
                no_cg=False, 
                fix_pos=False,
                n_kk_convs: int = 0,
                n_kk_heads: int = 4):
        super().__init__()

        if kp_rad != 0 and k_closest != 0:
            raise ValueError('one of kp_rad and kp_closest can be zero but not both')

        self.n_convs = n_convs
        self.n_keypoints = n_keypoints
        self.out_n_node_feat = out_n_node_feat
        self.kp_feat_scale = kp_feat_scale
        self.kp_pos_norm = out_n_node_feat**0.5
        self.k_closest = k_closest
        self.kp_rad = kp_rad
        self.no_cg = no_cg
        self.fix_pos = fix_pos
        self.use_sameres_feat = use_sameres_feat
        self.message_norm = message_norm

        self.graph_cutoffs = graph_cutoffs
        
        if self.use_sameres_feat:
            n_rr_conv_edge_feat = 1
        else:
            n_rr_conv_edge_feat = 0

        self.rec_convs = []

        for i in range(self.n_convs):
            if i == 0 and self.n_convs == 1: # first and only convolutional layer
                in_size = in_n_node_feat
                hidden_size = hidden_n_node_feat
                out_size = out_n_node_feat
            elif i == 0 and self.n_convs != 1: # first but not the only convolutional layer
                in_size = in_n_node_feat
                hidden_size = hidden_n_node_feat
                out_size = hidden_n_node_feat
            elif i == self.n_convs - 1 and self.n_convs != 1: # last but not the only convolutional layer
                in_size = hidden_n_node_feat
                hidden_size = hidden_n_node_feat
                out_size = out_n_node_feat
            else: # layers that are neither the first nor last layer 
                in_size = hidden_n_node_feat
                hidden_size = hidden_n_node_feat
                out_size = hidden_n_node_feat

            self.rec_convs.append( 
                ReceptorConv(in_size=in_size, 
                             hidden_size=hidden_size, 
                             out_size=out_size, 
                             use_tanh=use_tanh, 
                             coords_range=coords_range, 
                             message_norm=message_norm,
                             norm=norm, 
                             fix_pos=fix_pos, 
                             edge_feat_size=n_rr_conv_edge_feat)
            )

        self.rec_convs = nn.ModuleList(self.rec_convs)

        if no_cg:
            raise NotImplementedError

        # embedding function for the mean node feature before keypoint position generation
        self.keypoint_embedding = nn.Sequential(
            nn.Linear(out_n_node_feat, out_n_node_feat*n_keypoints),
            nn.SiLU()
        )

        self.rec_kp_conv = RecKeyConv(in_feats=self.out_n_node_feat, out_feats=out_n_node_feat, n_keypoints=self.n_keypoints, fix_pos=fix_pos, num_heads=1, k_closest=self.k_closest, kp_rad=kp_rad, norm=norm)

        self.n_kk_convs = n_kk_convs
        if self.n_kk_convs > 0:

            kk_convs = []
            for conv_idx in range(self.n_kk_convs):
                if conv_idx == 0:
                    pre_norm = False
                else:
                    pre_norm = True

                kk_convs.append(KeyKeyConv(in_feats=out_size, out_feats=out_size, num_heads=n_kk_heads, pre_norm=pre_norm))
            self.kk_convs = nn.ModuleList(kk_convs)


    def forward(self, g: dgl.DGLGraph, kp_batch_idx: torch.Tensor):

        x = g.nodes['rec'].data['x_0']
        h = g.nodes['rec'].data['h_0']
        batch_size = g.batch_size

        if self.use_sameres_feat:
            rec_edge_feat = g.edges['rr'].data['same_res']
        else:
            rec_edge_feat = None

        # compute rec_batch_idx, the batch index of every receptor atom
        rec_batch_idx = torch.arange(batch_size, device=g.device).repeat_interleave(g.batch_num_nodes('rec'))

        # compute normalization factor, z, for receptor-receptor message passing
        if self.message_norm == 0:
            z_rr = g.batch_num_edges(etype='rr') / g.batch_num_nodes('rec')
            z_rr = z_rr[rec_batch_idx].view(-1, 1)
        else:
            z_rr = self.message_norm

        # do equivariant message passing over nodes
        for conv_layer in self.rec_convs:
            h, x = conv_layer(g, node_feat=h, coord_feat=x, edge_feat=rec_edge_feat, z=z_rr)

        # record learned positions and features
        # TODO: is this necessary??
        g.nodes['rec'].data['x'] = x
        g.nodes['rec'].data['h'] = h

        if self.no_cg:
            raise NotImplementedError
            # TODO: remove keypoint nodes, add in n_keypoints = n_rec_atoms and set positions and features to the x and h we just obtained
            return g
        
        # compute the mean feature of the receptor
        mean_rec_feat = dgl.readout_nodes(g, 'h', op='mean', ntype='rec') # (batch_size, hidden_size)

        # initialize keypoint features
        init_kp_features = self.keypoint_embedding(mean_rec_feat)
        g.nodes['kp'].data['h_0'] = rearrange(init_kp_features, 'b (k d) -> (b k) d', d=self.out_n_node_feat, k=self.n_keypoints)

        # apply rec->kp graph attention convolution
        kp_pos, kp_feat = self.rec_kp_conv(g, kp_batch_idx)

        # assign keypoint positions and features
        g.nodes['kp'].data['h_0'] = kp_feat
        g.nodes['kp'].data['x_0'] = kp_pos

        # get batch info 
        batch_num_nodes, batch_num_edges = get_batch_info(g)

        # add keypoint-keypoint edges
        kk_edges = radius_graph(x=kp_pos, r=self.graph_cutoffs['kk'], batch=kp_batch_idx, max_num_neighbors=100)
        g.add_edges(kk_edges[0], kk_edges[1], etype='kk')

        # get number of keypoint-keypoint edges in each batch
        batch_num_edges[('kp', 'kk', 'kp')] = get_edges_per_batch(kk_edges[0], batch_size, kp_batch_idx)

        g.set_batch_num_nodes(batch_num_nodes)
        g.set_batch_num_edges(batch_num_edges)
        
        # do keypoint-keypoint convolutions if specified
        if self.n_kk_convs > 0:
            for conv in self.kk_convs:
                kp_feat = conv(g)
                g.nodes['kp'].data['h_0'] = kp_feat

        return g



