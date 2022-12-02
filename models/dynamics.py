import torch
import torch.nn as nn
from typing import Dict, List
import dgl.function as fn
import dgl

# TODO: rewrite EGNNConv for the receptor encoder
# TODO: should we do a sign embedding of distances? no, they don't use it in our benchmark

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


class LigRecEGNN(nn.Module):

    def __init__(self, n_layers, in_size, hidden_size, out_size):
        super().__init__()

        self.n_layers = n_layers
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.conv_layers: List[LigRecConv] = []
        for i in range(self.n_layers):
            if i == 0 and self.n_layers == 1: # first and only convolutional layer
                layer_in_size = in_size
                layer_hidden_size = hidden_size
                layer_out_size = out_size
            elif i == 0 and self.n_layers != 1: # first but not the only convolutional layer
                layer_in_size = in_size
                layer_hidden_size = hidden_size
                layer_out_size = hidden_size
            elif i == self.n_layers - 1 and self.n_layers != 1: # last but not the only convolutional layer
                layer_in_size = hidden_size
                layer_hidden_size = hidden_size
                layer_out_size = out_size
            else: # layers that are neither the first nor last layer 
                layer_in_size = hidden_size
                layer_hidden_size = hidden_size
                layer_out_size = hidden_size

            self.conv_layers.append( 
                LigRecConv(in_size=layer_in_size, hidden_size=layer_hidden_size, out_size=layer_out_size).double() 
            )

    def forward(self, graph):

        node_coords = graph.ndata['x_0']
        node_features = graph.ndata['h_0']

        # do equivariant message passing on the heterograph
        h, x = self.conv_layers[0](graph, node_features, node_coords)
        if len(self.conv_layers) > 1:
            for conv_layer in self.conv_layers[1:]:
                h, x = conv_layer(graph, h, x)

        return h, x



class LigRecDynamics(nn.Module):

    def __init__(self, atom_nf, rec_nf, n_layers=4, joint_nf=32, hidden_nf=256, act_fn=nn.SiLU):
        super().__init__()    

    
        self.lig_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn(),
            nn.Linear(2 * atom_nf, joint_nf)
        )

        self.lig_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf),
            act_fn(),
            nn.Linear(2 * atom_nf, atom_nf)
        )

        self.rec_encoder = nn.Sequential(
            nn.Linear(rec_nf, 2 * rec_nf),
            act_fn(),
            nn.Linear(2 * rec_nf, joint_nf)
        )

        # we add +1 to the input feature size for the timestep
        self.egnn = LigRecEGNN(n_layers=n_layers, in_size=joint_nf+1, hidden_size=hidden_nf, out_size=joint_nf)


    def forward(self, lig_pos, lig_feat, rec_pos, rec_feat, timestep):
        # inputs: ligand/receptor positions/features, timestep
        # outputs: predicted noise

        lig_feat = list(lig_feat)
        rec_feat = list(rec_feat)


        # encode lig/rec features
        for i in range(len(lig_feat)):
            lig_feat[i] = self.lig_encoder(lig_feat[i])
            rec_feat[i] = self.rec_encoder(rec_feat[i])

    
        # add timestep to node features
        lig_feat_time = []
        rec_feat_time = []
        for i in range(len(lig_feat)):
            t_reshaped = timestep[i].repeat(lig_feat[i].shape[0]).view(-1,1)
            lig_feat_time.append( torch.cat([lig_feat[i], t_reshaped]) )

            t_reshaped = timestep[i].repeat(rec_feat[i].shape[0]).view(-1,1)
            rec_feat_time.append( torch.cat([rec_feat[i], t_reshaped]) )

        # construct heterograph
        graph = self.make_graph(lig_pos, lig_feat_time, rec_pos, rec_feat_time)

        # pass through LigRecEGNN
        h, x = self.egnn(graph)

        # decode lig features

    def make_graph(lig_pos, lig_feat, rec_pos, rec_feat):

        # note that all arguments except timestep are tuples of length batch_size containing
        # the values for each datapoint in the batch

        k_rl = 4 # receptor keypoints have edges to the k_rl nearest ligand atoms
        
        graphs = []
        for i in range(len(lig_pos)):

            # create graph containing just ligand-ligand edges
            lig_graph = dgl.knn_graph(lig_pos[i], k=4, algorithm="bruteforce-blas", dist='euclidean')

            # find edges for rec -> lig conections
            rl_dist = torch.cdist(rec_pos[i], lig_pos[i]) # distance between every receptor keypoint and every ligand atom
            topk_dist, topk_idx = torch.topk(rl_dist, k=4, dim=1, largest=False) # get k closest ligand atoms to each receptor atom

            # get list of rec -> ligand edges
            n_rec_nodes = rec_pos[i].shape[0]
            src_nodes = torch.repeat_interleave(torch.arange(n_rec_nodes), repeats=k_rl)
            dst_nodes = topk_idx.flatten()

            # create heterograph
            graph_data = {

            ('lig', 'll', 'lig'): lig_graph.edges(),

            ('rec', 'rl', 'lig'): (src_nodes, dst_nodes)

            }

            g = dgl.heterograph(graph_data)
            g.nodes['lig'].data['x_0'] = lig_pos[i]
            g.nodes['lig'].data['h_0'] = lig_feat[i]
            g.nodes['rec'].data['x_0'] = rec_pos[i]
            g.nodes['rec'].data['h_0'] = rec_feat[i]
            graphs.append(g)
        
        batched_graph = dgl.batch(graphs)
        return batched_graph