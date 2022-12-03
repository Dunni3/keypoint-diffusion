from errno import E2BIG
import torch
import dgl
import torch.nn as nn
from dgl.nn import EGNNConv
from typing import List

class ReceptorEncoder(nn.Module):

    def __init__(self, n_egnn_convs: int, n_keypoints: int, in_n_node_feat: int, hidden_n_node_feat: int, out_n_node_feat: int):
        super().__init__()

        self.n_egnn_convs = n_egnn_convs
        self.n_keypoints = n_keypoints
        self.out_n_node_feat = out_n_node_feat

        self.egnn_convs = []

        # TODO: the DGL EGNN implementation uses two-layer MLPs - does the EDM paper use 1 or 2?
        # TODO: the EDM paper has a skip connection in the update of node features (which is not specified in model equations)
        # - does DGL have this as well?
        for i in range(self.n_egnn_convs):
            if i == 0 and self.n_egnn_convs == 1: # first and only convolutional layer
                in_size = in_n_node_feat
                hidden_size = hidden_n_node_feat
                out_size = out_n_node_feat
            elif i == 0 and self.n_egnn_convs != 1: # first but not the only convolutional layer
                in_size = in_n_node_feat
                hidden_size = hidden_n_node_feat
                out_size = hidden_n_node_feat
            elif i == self.n_egnn_convs - 1 and self.n_egnn_convs != 1: # last but not the only convolutional layer
                in_size = hidden_n_node_feat
                hidden_size = hidden_n_node_feat
                out_size = out_n_node_feat
            else: # layers that are neither the first nor last layer 
                in_size = hidden_n_node_feat
                hidden_size = hidden_n_node_feat
                out_size = hidden_n_node_feat

            self.egnn_convs.append( 
                EGNNConv(in_size=in_size, hidden_size=hidden_size, out_size=out_size)
            )

            self.eqv_keypoint_query_fn = nn.Linear(in_features=out_n_node_feat, out_features=out_n_node_feat*n_keypoints)
            self.eqv_keypoint_key_fn = nn.Linear(in_features=out_n_node_feat, out_features=out_n_node_feat*n_keypoints)

            # self.inv_keypoint_query_fn = nn.Linear(in_features=out_n_node_feat, out_features=out_n_node_feat*n_keypoints)
            # self.inv_keypoint_key_fn = nn.Linear(in_features=out_n_node_feat, out_features=out_n_node_feat*n_keypoints)

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
            mean_node_feature = dgl.mean_nodes(graph, feat='h').view(1, self.out_n_node_feat) # shape (1, n_node_features)
            eqv_queries = self.eqv_keypoint_key_fn(mean_node_feature).view(self.n_keypoints, self.out_n_node_feat) # shape (n_attn_heads, n_node_feautres)
            eqv_keys = self.eqv_keypoint_query_fn(graph.ndata['h']).view(-1, self.n_keypoints, self.out_n_node_feat) # (n_nodes, n_attn_heads, n_node_features)
            eqv_att_logits = torch.einsum('ijk,jk->ji', eqv_keys, eqv_queries) # (n_attn_heads, n_nodes)
            eqv_att_weights = torch.softmax(eqv_att_logits, dim=1)
            kp_pos = eqv_att_weights @ graph.ndata['x'] # (n_keypoints, 3)
            keypoint_positions.append(kp_pos)

            # in my first attempt, i computed invariant keypoints by a mechanism similar to that used 
            # for the equivariant keypoints. then i decided on a simpler approach:
            # each invariant keypoint will be a weighted sum of the learned feature vectors for all receptor atoms
            # the weights in this weighted sum will soft-maxes over the distance between the key point position and all receptor atoms
            # the next block of commented code is the first method, and the following un-commented block of code is the second method

            # compute invariant keypoints
            # inv_queries = self.inv_keypoint_key_fn(mean_node_feature).view(self.n_keypoints, self.out_n_node_feat) # shape (n_attn_heads, n_node_feautres)
            # inv_keys = self.inv_keypoint_query_fn(graph.ndata['h']).view(-1, self.n_keypoints, self.out_n_node_feat) # (n_nodes, n_attn_heads, n_node_features)
            # inv_att_logits = torch.einsum('ijk,jk->ji', inv_keys, inv_queries) # (n_attn_heads, n_nodes)
            # inv_att_weights = torch.softmax(inv_att_logits, dim=1)
            # kp_feat = inv_att_weights @ graph.ndata['h']
            # keypoint_features.append(kp_feat)

            # compute distance between keypoints and binding pocket points
            kp_dist = torch.cdist(kp_pos, graph.ndata['x_0'])
            kp_feat_weights = torch.softmax(kp_dist, dim=1) # (n_keypoints, n_pocket_atoms)
            kp_feat = kp_feat_weights @ graph.ndata["h"]
            keypoint_features.append(kp_feat)

        return keypoint_positions, keypoint_features