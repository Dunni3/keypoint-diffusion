import torch
import torch.nn as nn
from typing import Dict
import dgl.function as fn


class LigRecConv(nn.Module):

    # this class was adapted from the DGL implementation of EGNN
    # original code: https://github.com/dmlc/dgl/blob/76bb54044eb387e9e3009bc169e93d66aa004a74/python/dgl/nn/pytorch/conv/egnnconv.py
    # I have extended the EGNN graph conv layer to operate on heterogeneous graphs containing containing receptor and ligand nodes

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super().__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        self.edge_types = ["ll", "rl"]

        # TODO: how many layers are in the MLPs of Hoogeboom/Arnescheung papers?
        # TODO: what activation function do they use in Hoogeboom? which should i use?

        # \phi_e^t
        self.edge_mlp: Dict[str, nn.Sequential] = {}
        for edge_type in self.edge_types:
            self.edge_mlp[edge_type] = nn.Sequential(
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

        # \phi_x^t
        self.coord_mlp: Dict[str, nn.Sequential] = {}
        for edge_type in self.edge_types:
            self.coord_mlp[edge_type] = nn.Sequential(
                # +1 for the radial feature: ||x_i - x_j||^2
                nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
                act_fn,
                nn.Linear(hidden_size, 1),
                act_fn,
            )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge-wise operations
        if self.edge_feat_size > 0:
            f = torch.cat(
                [
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["dij"],
                    edges.data["a"],
                ],
                dim=-1,
            )
        else:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["dij"]], dim=-1
            )

        # get edge type
        edge_type = edges.canonical_etype[1]

        # compute feature messages
        msg_h = self.edge_mlp[edge_type](f)

        # compute coordinate messages
        msg_x = self.coord_mlp[edge_type](f) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            Heterogeneous graph with ligand and receptor nodes.
        node_feat : Dict[str, torch.Tensor]
            Ligand and receptor node features
        coord_feat : Dict[str, torch.Tensor]
            Ligand and receptor coordinate features
        edge_feat : torch.Tensor, optional
            Edge features. Not presently implemented. 

        Returns
        -------
        node_feat_out : Dict[str, torch.Tensor]
        coord_feat_out: Dict[str, torch.Tensor]
        """
        assert graph.etypes == self.edge_types
        with graph.local_scope():
            # node feature
            graph.ndata["h"] = node_feat
            # coordinate feature
            graph.ndata["x"] = coord_feat

            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
                

            # compute displacement vector between nodes for all edges
            # TODO: this is u_sub_v ... should we do v_sub_u? which way is information flowing? does this 
            # affect how I construct edges in my initial graph??
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))

            # compute euclidean distance across all edges
            for etype in graph.etypes:
                graph.apply_edges(
                    lambda edges: {'dij': 
                    torch.linalg.vector_norm(edges.data['x_diff'], dim=1).unsqueeze(-1)},
                    etype=etype)


            # normalize displacement vectors to unit length
            for etype in graph.etypes:
                graph.apply_edges(
                    lambda edges: {'x_diff': edges.data['x_diff'] / (edges.data['dij'] + 1e-9)},
                    etype=etype)


            # compute messages and store them on every edge
            for etype in graph.etypes:
                graph.apply_edges(self.message, etype=etype)

            # aggregating messages from all edges
            graph.update_all(fn.copy_e("msg_x", "m"), fn.sum("m", "x_neigh"))
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))


            # get aggregated messages
            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]


            # compute updated features/coordinates
            # note that the receptor features/coordinates are not updated
            node_mlp_input = torch.cat([node_feat['lig'], h_neigh['lig']], dim=-1)
            h = {'lig': node_feat['lig'] + self.node_mlp(node_mlp_input),
                 'rec': node_feat['rec']}


            x = {'lig': coord_feat['lig'] + x_neigh['lig'],
                 'rec': coord_feat['rec']}

            return h, x

