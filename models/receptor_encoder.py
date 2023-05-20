from errno import E2BIG
import torch
import dgl
import torch.nn as nn
# from dgl.nn import EGNNConv
from typing import List
from einops import rearrange

import dgl.function as fn

class EGNNConv(nn.Module):
    # this is adapted from the EGNN implementation in DGL

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, use_tanh=True, coords_range=10, message_norm=1):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()
        self.use_tanh = use_tanh
        self.coords_range = coords_range
        self.message_norm = message_norm

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


        # \phi_x
        coord_output_layer = nn.Linear(hidden_size, 1, bias=False)
        nn.init.xavier_uniform_(coord_output_layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
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
        if self.use_tanh:
            msg_x = torch.tanh( self.coord_mlp(msg_h) ) * edges.data["x_diff"] * self.coords_range
        else:
            msg_x = self.coord_mlp(msg_h) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
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
        with graph.local_scope():
            # node feature
            graph.ndata["h"] = node_feat
            # coordinate feature
            graph.ndata["x"] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["radial"] = (
                graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1)
            )
            # normalize coordinate difference
            graph.edata["x_diff"] = graph.edata["x_diff"] / (
                graph.edata["radial"].sqrt() + 1e-30
            )
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh"))
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))

            h_neigh, x_neigh = graph.ndata["h_neigh"]/self.message_norm, graph.ndata["x_neigh"]/self.message_norm

            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh

            return h, x

class KeypointMHA(nn.Module):

    def __init__(self, n_heads: int, in_dim: int, hidden_dim: int, out_dim: int, act_fn = nn.SiLU):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.key_fn = nn.Linear(in_dim, hidden_dim*n_heads, bias=False)
        self.query_fn = nn.Linear(in_dim, hidden_dim*n_heads, bias=False)
        self.val_fn = nn.Linear(in_dim, hidden_dim*n_heads, bias=False)

        self.out_map = nn.Linear(hidden_dim*n_heads, out_dim, bias=False)

        self.kq_norm = self.hidden_dim**0.5
        self.dist_norm = 3**0.5
        self.att_norm = 2**0.5

    def forward(self, kp_feat, kp_dist, rec_atom_feats):

        # kp_feat (n_keypoints, in_dim)
        # kp_dist (n_keypoints, n_receptor_atoms)
        # rec_atom_feats (n_rec_atoms, in_dim)
        n_keypoints = kp_feat.shape[0]
        n_rec_atoms = rec_atom_feats.shape[0]

        # generate keys and values from kp_feat
        keys = self.key_fn(rec_atom_feats).view(self.n_heads, n_rec_atoms, self.hidden_dim) # (n_heads, n_rec_atoms, hidden_dim)
        values = self.val_fn(rec_atom_feats).view(self.n_heads, n_rec_atoms, self.hidden_dim) # (n_heads, n_rec_atoms, hidden_dim)

        # generate queries
        queries = self.query_fn(kp_feat).view(self.n_heads, n_keypoints, self.hidden_dim) # (n_heads, n_keypoints, hidden_dim)

        # compute dot-product of keys/queries for all pairs of keypoints
        qk_dot = torch.einsum('hkd,had->hka' , queries, keys) # (n_heads, n_keypoints, n_rec_atoms)

        # compute pre-softmax attention matrix, then take its softmax
        att_mat = (kp_dist.unsqueeze(0)/self.dist_norm + qk_dot/self.kq_norm)/self.att_norm # (n_heads, n_keypoints, n_rec_atoms)
        att_weights = torch.softmax(att_mat, dim=2)

        # multiply attention weights by values
        updated_values = torch.einsum('hka,had->hkd', att_weights, values) # (n_heads, n_keypoints, hidden_dim)

        # combine n_heads and hidden_dim dimensons
        kp_wise_values = rearrange(updated_values, 'h k d -> k (h d)') # (n_keypoints, n_heads*hidden_dim)

        updated_kp_feat = self.out_map(kp_wise_values)
        return updated_kp_feat

class ReceptorEncoder(nn.Module):

    def __init__(self, n_convs: int = 6, n_keypoints: int = 10, in_n_node_feat: int = 13, 
        hidden_n_node_feat: int = 256, out_n_node_feat: int = 256, use_tanh=True, coords_range=10, kp_feat_scale=1,
        use_keypoint_feat_mha: bool = False, feat_mha_heads=5, message_norm=1, k_closest: int = 0):
        super().__init__()

        self.n_convs = n_convs
        self.n_keypoints = n_keypoints
        self.out_n_node_feat = out_n_node_feat
        self.kp_feat_scale = kp_feat_scale
        self.kp_pos_norm = out_n_node_feat**0.5
        self.k_closest = k_closest

        # TODO: should there be a position-wise MLP after graph convolution?
        # TODO: this model is written to use the same output dimension from the graph message passing as for the keypoint feature attention mechianism -- this is an articifical constraint

        self.egnn_convs = []

        # TODO: the DGL EGNN implementation uses two-layer MLPs - does the EDM paper use 1 or 2?
        # TODO: the EDM paper has a skip connection in the update of node features (which is not specified in model equations)
        # - does DGL have this as well?
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

            self.egnn_convs.append( 
                EGNNConv(in_size=in_size, hidden_size=hidden_size, out_size=out_size, use_tanh=use_tanh, coords_range=coords_range, message_norm=message_norm)
            )

        self.egnn_convs = nn.ModuleList(self.egnn_convs)


        # embedding function for the mean node feature before keypoint position generation
        self.node_feat_embedding = nn.Sequential(
            nn.Linear(out_n_node_feat, out_n_node_feat),
            nn.SiLU()
        )


        # query and key functions for keypoint position generation
        self.eqv_keypoint_query_fn = nn.Linear(in_features=out_n_node_feat, out_features=out_n_node_feat*n_keypoints, bias=False)
        self.eqv_keypoint_key_fn = nn.Linear(in_features=out_n_node_feat, out_features=out_n_node_feat*n_keypoints, bias=False)


        # keypoint-wise MLP applied to keypoint features when they are first
        # generated as weighted averages over receptor atom features
        self.kp_wise_mlp = nn.Sequential(
            nn.Linear(out_n_node_feat+self.k_closest, out_n_node_feat*2),
            nn.SiLU(),
            nn.Linear(out_n_node_feat*2, out_n_node_feat),
            nn.SiLU()
        )

        self.use_keypoint_feat_mha = use_keypoint_feat_mha
        if self.use_keypoint_feat_mha:
            self.keypoint_feat_mha = KeypointMHA(n_heads=feat_mha_heads, in_dim=out_n_node_feat, hidden_dim=out_n_node_feat, out_dim=out_n_node_feat)
            self.post_mha_dense_layer = nn.Sequential(
                nn.Linear(out_n_node_feat, out_n_node_feat*4),
                nn.SiLU(),
                nn.Linear(out_n_node_feat*4, out_n_node_feat),
                nn.SiLU()
            )


    def forward(self, rec_graph: dgl.DGLGraph):
        node_positions = rec_graph.ndata['x_0']
        node_features = rec_graph.ndata['h_0']

        # do equivariant message passing over nodes
        h, x = self.egnn_convs[0](rec_graph, node_features, node_positions)
        if len(self.egnn_convs) > 1:
            for conv_layer in self.egnn_convs[1:]:
                h, x = conv_layer(rec_graph, h, x)

        # record learned positions and features
        rec_graph.ndata['x'] = x
        rec_graph.ndata['h'] = h

        # TODO: apply atom-wise MLP in h?
        # not necessary because sum of messages goes through an MLP

        keypoint_positions = []
        keypoint_features = []
        for graph_idx, graph in enumerate(dgl.unbatch(rec_graph)):

            # compute equivariant keypoints
            mean_node_feature = self.node_feat_embedding(graph.ndata['h']).mean(dim=0) # shape (1, n_node_features)
            eqv_queries = self.eqv_keypoint_key_fn(mean_node_feature).view(self.n_keypoints, self.out_n_node_feat) # shape (n_attn_heads, n_node_feautres)
            eqv_keys = self.eqv_keypoint_query_fn(graph.ndata['h']).view(-1, self.n_keypoints, self.out_n_node_feat) # (n_nodes, n_attn_heads, n_node_features)
            eqv_att_logits = torch.einsum('ijk,jk->ji', eqv_keys, eqv_queries) # (n_attn_heads, n_nodes)
            eqv_att_weights = torch.softmax(eqv_att_logits/self.kp_pos_norm, dim=1)
            kp_pos = eqv_att_weights @ graph.ndata['x'] # (n_keypoints, 3)
            keypoint_positions.append(kp_pos)

            # compute distance between keypoints and binding pocket points
            kp_dist = torch.cdist(kp_pos, graph.ndata['x_0'])

            # get keypoint features as softmax over distance to receptor atoms
            if self.k_closest == 0:
                kp_feat_weights = torch.softmax(-1.0*kp_dist*self.kp_feat_scale, dim=1) # (n_keypoints, n_pocket_atoms)
                kp_feat = kp_feat_weights @ graph.ndata["h"]
            else:
                top_vals, top_idx = torch.topk(kp_dist, k=self.k_closest, dim=1, largest=False)
                kp_feat = graph.ndata["h"][top_idx, :].mean(dim=1)
                kp_feat = torch.concat([kp_feat, top_vals], dim=1)

            # apply keypoint-wise MLP before input to attention mechanism
            kp_feat = self.kp_wise_mlp(kp_feat)

            # update keypoint features by receptor-kp cross-attention
            if self.use_keypoint_feat_mha:
                kp_feat = kp_feat + self.keypoint_feat_mha(kp_feat, kp_dist, graph.ndata["h"])
                kp_feat = self.post_mha_dense_layer(kp_feat)

            keypoint_features.append(kp_feat)

        return keypoint_positions, keypoint_features


