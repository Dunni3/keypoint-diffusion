from errno import E2BIG
import torch
import dgl
import torch.nn as nn
# from dgl.nn import EGNNConv
from typing import List

import dgl.function as fn

class EGNNConv(nn.Module):
    # this is adapted from the EGNN implementation in DGL

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, use_tanh=True, coords_range=10):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()
        self.use_tanh = use_tanh
        self.coords_range = coords_range

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

            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh

            return h, x

class KeypointMHA(nn.Module):

    def __init__(self, n_heads: int, in_dim: int, hidden_dim: int, act_fn = nn.SiLU):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim

        self.key_fn = nn.Linear(in_dim, hidden_dim*n_heads, bias=False)
        self.query_fn = nn.Linear(in_dim, hidden_dim*n_heads, bias=False)
        self.val_fn = nn.Linear(in_dim, hidden_dim*n_heads, bias=False)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim*n_heads, hidden_dim*2),
            act_fn(),
            nn.Linear(hidden_dim*2, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, in_dim),
            act_fn()
        )

        self.kq_norm = self.hidden_dim**0.5
        self.dist_norm = 0.5
        self.att_norm = 2**0.5

    def forward(self, kp_pos, kp_feat):
        # kp_pos (batch_size, n_keypoints, 3)
        # kp_feat (batch_size, n_keypoints, in_dim)
        batch_size, n_keypoints, _ = kp_pos.shape

        # compute pairwise keypoint distances
        dist_mat = torch.cdist(kp_pos, kp_pos).unsqueeze(1) # (batch_size, 1, n_keypoints, n_keypoints)

        # compute keys, queries, and values
        keys = self.key_fn(kp_feat).view(batch_size, self.n_heads, n_keypoints, self.hidden_dim) # (batch_size, n_heads, n_keypoints, hidden_dim)
        queries = self.query_fn(kp_feat).view(batch_size, self.n_heads, n_keypoints, self.hidden_dim) # (batch_size, n_heads, n_keypoints, hidden_dim)
        values = self.val_fn(kp_feat).view(batch_size, self.n_heads, n_keypoints, self.hidden_dim) # (batch_size, n_heads, n_keypoints, hidden_dim)

        # compute dot-product of keys/queries for all pairs of keypoints
        kq_dot = torch.einsum('bhkd,bhkd->bhkk' , keys, queries) # (batch_size, n_heads, n_keypoints, n_keypoints)

        # compute pre-softmax attention matrix, then take its softmax
        att_mat = (dist_mat/self.dist_norm + kq_dot/self.kq_norm)/self.att_norm # (batch_size, n_heads, n_keypoints, n_keypoints)
        att_weights = torch.softmax(att_mat, dim=3)
        
        # multiply attention weights by values
        updated_values = torch.einsum('bhkk,bhkd->hdbk', att_weights, values) # (n_heads, hidden_dim, batch_size, n_keypoints)

        # collapse n_heads and hidden_dim on to each other
        updated_values = updated_values.view(self.n_heads*self.hidden_dim, batch_size, n_keypoints).transpose(1,2,0) # (batch_size, n_keypoints, n_heads*hidden_dim)

        output = self.out_mlp(updated_values) # (batch_size, n_keypoints, in_dim)

        return output

class ReceptorEncoder(nn.Module):

    def __init__(self, n_convs: int = 6, n_keypoints: int = 10, in_n_node_feat: int = 13, 
        hidden_n_node_feat: int = 256, out_n_node_feat: int = 256, use_tanh=True, coords_range=10, kp_feat_scale=1,
        keypoint_postprocess: str = None, post_n_heads=5, post_hidden_dim=256):
        super().__init__()

        self.n_convs = n_convs
        self.n_keypoints = n_keypoints
        self.out_n_node_feat = out_n_node_feat
        self.kp_feat_scale = kp_feat_scale
        self.kp_pos_norm = out_n_node_feat**0.5
        
        self.keypoint_postprocess = keypoint_postprocess

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
                EGNNConv(in_size=in_size, hidden_size=hidden_size, out_size=out_size, use_tanh=use_tanh, coords_range=coords_range)
            )

            self.egnn_convs = nn.ModuleList(self.egnn_convs)

            self.node_feat_embedding = nn.Sequential(
                nn.Linear(out_n_node_feat, out_n_node_feat),
                nn.LeakyReLU()
            )

            self.eqv_keypoint_query_fn = nn.Linear(in_features=out_n_node_feat, out_features=out_n_node_feat*n_keypoints)
            self.eqv_keypoint_key_fn = nn.Linear(in_features=out_n_node_feat, out_features=out_n_node_feat*n_keypoints)

            if self.keypoint_postprocess == "attention":
                self.postprocess_layer = KeypointMHA(n_heads=post_n_heads, in_dim=out_n_node_feat, hidden_dim=post_hidden_dim)


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

        # are the graphs returned in the same order which mean features are caclculated for? yes!
        graphs = dgl.unbatch(rec_graph)

        keypoint_positions = []
        keypoint_features = []
        for graph_idx, graph in enumerate(dgl.unbatch(rec_graph)):

            # compute equivariant keypoints
            # embedded_node_features = self.node_feat_embedding(graph.ndata['h']) # shape (n_pocket_atoms, n_node_features)
            mean_node_feature = self.node_feat_embedding(graph.ndata['h']).mean(dim=0) # shape (1, n_node_features)
            eqv_queries = self.eqv_keypoint_key_fn(mean_node_feature).view(self.n_keypoints, self.out_n_node_feat) # shape (n_attn_heads, n_node_feautres)
            eqv_keys = self.eqv_keypoint_query_fn(graph.ndata['h']).view(-1, self.n_keypoints, self.out_n_node_feat) # (n_nodes, n_attn_heads, n_node_features)
            eqv_att_logits = torch.einsum('ijk,jk->ji', eqv_keys, eqv_queries) # (n_attn_heads, n_nodes)
            eqv_att_weights = torch.softmax(eqv_att_logits/self.kp_pos_norm, dim=1)
            kp_pos = eqv_att_weights @ graph.ndata['x'] # (n_keypoints, 3)
            keypoint_positions.append(kp_pos)

            # compute distance between keypoints and binding pocket points
            kp_dist = torch.cdist(kp_pos, graph.ndata['x_0'])
            kp_feat_weights = torch.softmax(kp_dist*self.kp_feat_scale, dim=1) # (n_keypoints, n_pocket_atoms)
            kp_feat = kp_feat_weights @ graph.ndata["h"]
            keypoint_features.append(kp_feat)

        if self.keypoint_postprocess is not None:
            kp_pos_stacked = torch.stack(keypoint_positions, dim=0)
            kp_feat_stacked = torch.stack(keypoint_features, dim=0)

        if self.keypoint_postprocess == "attention":
            kp_feat_stacked = self.postprocess_layer(kp_pos_stacked, kp_feat_stacked) 
            keypoint_features = list(torch.unbind(kp_feat_stacked, dim=0))

        return keypoint_positions, keypoint_features


