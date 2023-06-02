import torch
import torch.nn as nn
from typing import Dict, List
import dgl.function as fn
import dgl
from torch_cluster import radius, radius_graph
from utils import get_batch_info, get_edges_per_batch

class LigRecConv(nn.Module):

    # this class was adapted from the DGL implementation of EGNN
    # original code: https://github.com/dmlc/dgl/blob/76bb54044eb387e9e3009bc169e93d66aa004a74/python/dgl/nn/pytorch/conv/egnnconv.py
    # I have extended the EGNN graph conv layer to operate on heterogeneous graphs containing containing receptor and ligand nodes

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, use_tanh=False, coords_range=10, update_kp_feat: bool = False, norm: bool = False):
        super().__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()
        self.use_tanh = use_tanh
        self.update_kp_feat = update_kp_feat
        self.norm = norm

        self.coords_range = coords_range

        if self.update_kp_feat:
            self.edge_types = ["ll", "kl", "lk", "kk"]
            self.updated_node_types = ['lig', 'kp']
        else:
            self.edge_types = ["ll", "kl"]
            self.updated_node_types = ['lig']

        # \phi_e^t
        self.edge_mlp = {}
        for edge_type in self.edge_types:
            self.edge_mlp[edge_type] = nn.Sequential(
                # +1 for the radial feature: ||x_i - x_j||^2
                nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
                act_fn,
                nn.Linear(hidden_size, hidden_size),
                act_fn,
            )
        self.edge_mlp = nn.ModuleDict(self.edge_mlp)

        self.soft_attention = {}
        for edge_type in self.edge_types:
            self.soft_attention[edge_type] = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.soft_attention = nn.ModuleDict(self.soft_attention)

        # \phi_h
        self.node_mlp = {}
        for ntype in self.updated_node_types:
            self.node_mlp[ntype] = nn.Sequential(
                nn.Linear(in_size + hidden_size, hidden_size),
                act_fn,
                nn.Linear(hidden_size, out_size),
            )
        self.node_mlp = nn.ModuleDict(self.node_mlp)

        # \phi_x^t
        self.coord_mlp = {}
        for edge_type in self.edge_types:
            layer = nn.Linear(hidden_size, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            self.coord_mlp[edge_type] = nn.Sequential(
                # +1 for the radial feature: ||x_i - x_j||^2
                nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
                act_fn,
                nn.Linear(hidden_size, hidden_size),
                act_fn,
                layer
            )
        self.coord_mlp = nn.ModuleDict(self.coord_mlp)

        self.layer_norm = {}
        for ntype in self.updated_node_types:
            if self.norm:
                self.layer_norm[ntype] = nn.LayerNorm(out_size)
            else:
                self.layer_norm[ntype] = nn.Identity()
        self.layer_norm = nn.ModuleDict(self.layer_norm)

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
        msg_h = msg_h*self.soft_attention[edge_type](msg_h)

        # compute coordinate messages
        if edge_type[1] in ["kk", "lk"]:
            msg_x = torch.zeros_like(edges.data["radial"])
        elif self.use_tanh:
            msg_x = torch.tanh( self.coord_mlp[edge_type](f) )* edges.data["x_diff"] * self.coords_range
        else:
            msg_x = self.coord_mlp[edge_type](f)*edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, graph: dgl.DGLHeteroGraph, node_feat: Dict[str, torch.Tensor], coord_feat: Dict[str, torch.Tensor], z_dict: Dict[str, torch.Tensor], edge_feat=None):
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
        with graph.local_scope():
            # node feature
            graph.ndata["h"] = node_feat
            # coordinate feature
            graph.ndata["x"] = coord_feat
            
            # edge feature
            if self.edge_feat_size > 0:
                raise NotImplementedError
                # assert edge_feat is not None, "Edge features must be provided."
                # graph.edata["a"] = edge_feat

            # compute displacement vector between nodes for all edges
            for etype in self.edge_types:
                graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"), etype=etype)

            # compute euclidean distance across all edges
            for etype in self.edge_types:
                graph.apply_edges(self.compute_dij, etype=etype)

            # normalize displacement vectors to unit length
            for etype in self.edge_types:
                graph.apply_edges(
                    lambda edges: {'x_diff': edges.data['x_diff'] / (edges.data['dij'] + 1)},
                    etype=etype)

            # compute messages and store them on every edge
            for etype in self.edge_types:
                graph.apply_edges(self.message, etype=etype)

            # aggregating messages from all edges
            x_update_dict = {}
            for etype in self.edge_types:
                x_update_dict[etype] = (fn.copy_e("msg_x", "m"), fn.sum("m", "x_neigh"))
            graph.multi_update_all(x_update_dict, cross_reducer='sum')

            h_update_dict = {}
            for etype in self.edge_types:
                h_update_dict[etype] = (fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))
            graph.multi_update_all(h_update_dict, cross_reducer='sum')

            # normalize messages
            for key in graph.ndata['h_neigh']:
                graph.ndata['h_neigh'][key] = graph.ndata['h_neigh'][key]/z_dict[key]

            for key in graph.ndata['x_neigh']:
                graph.ndata['x_neigh'][key] = graph.ndata['x_neigh'][key]/z_dict[key]

            # get aggregated messages
            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

            # compute updated features/coordinates
            # note that updates for kp positions will always be 0
            h = {}
            x = {}
            for ntype in self.updated_node_types:
                node_mlp_input = torch.concatenate([ node_feat[ntype], h_neigh[ntype] ], dim=1)
                new_node_feat = node_feat[ntype] + self.node_mlp[ntype](node_mlp_input)
                new_node_feat = self.layer_norm[ntype](new_node_feat)
                h[ntype] = new_node_feat
                x[ntype] = coord_feat[ntype] + x_neigh[ntype]

            return h, x

    def compute_dij(self, edges):
        dij = torch.linalg.vector_norm(edges.data['x_diff'], dim=1).unsqueeze(-1)
        # when the distance between two points is 0 (or very small)
        # the backwards pass through the comptuation of the distance
        # requires a backwards pass through torch.sqrt() which produces nan gradients
        # with torch.no_grad():
        #     dij.clamp_(min=1e-2)
        return {'dij': dij}



class LigRecEGNN(nn.Module):

    def __init__(self, n_layers, in_size, hidden_size, out_size, use_tanh=False, message_norm=1, update_kp_feat: bool = False, norm: bool = False):
        super().__init__()

        self.n_layers = n_layers
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.update_kp_feat = update_kp_feat
        self.norm = norm
        self.message_norm = message_norm

        if update_kp_feat:
            self.edge_types = ["ll", "kl", "lk", "kk"]
            self.updated_node_types = ['lig', 'kp']
        else:
            self.edge_types = ["ll", "kl"]
            self.updated_node_types = ['lig']

        self.conv_layers = []
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
                LigRecConv(in_size=layer_in_size, hidden_size=layer_hidden_size, out_size=layer_out_size, use_tanh=use_tanh, update_kp_feat=update_kp_feat, norm=norm)
            )

            self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, graph: dgl.DGLHeteroGraph, lig_batch_idx, kp_batch_idx):

        h = {}
        x = {}
        for ntype in ['lig', 'kp']:
            h[ntype] = graph.nodes[ntype].data['h_0']
            x[ntype] = graph.nodes[ntype].data['x_0']


        # compute z, the normalization factor for messages passed on the graph, for each node type that is updated
        # we choose z to be the average in-degree of nodes being update, across all node types that are updated.
        z_dict = {}
        batch_dict = { 'lig': lig_batch_idx, 'kp': kp_batch_idx}
        for ntype in self.updated_node_types:
            # TODO: possibly faster to do one sum call with torch.sum
            if self.message_norm == 0:
                z_dict[ntype] = torch.stack([graph.batch_num_edges(etype) for etype in self.edge_types if etype[-1] == ntype[0] ], dim=0).sum(dim=0) / graph.batch_num_nodes(ntype)
                z_dict[ntype] = z_dict[ntype][ batch_dict[ntype] ].view(-1, 1) + 1
            else:
                z_dict[ntype] = self.message_norm

        # do equivariant message passing on the heterograph
        for layer in self.conv_layers:
            if 'kp' not in h: # this occurs when update_kp_feat = False
                h['kp'] = graph.nodes['kp'].data['h_0']
                x['kp'] = graph.nodes['kp'].data['x_0']
            h,x = layer(graph, h, x, z_dict)

        return h['lig'], x['lig']



class LigRecDynamics(nn.Module):

    def __init__(self, atom_nf, rec_nf, n_layers=4, hidden_nf=255, act_fn=nn.SiLU, use_tanh=False, message_norm=1, no_cg: bool = False,
                 n_keypoints: int = 20, graph_cutoffs: dict = {}, update_kp_feat: bool = False, norm: bool = False):
        super().__init__()

        self.no_cg = no_cg    
        self.n_keypoints = n_keypoints
        self.graph_cutoffs = graph_cutoffs
        self.update_kp_feat = update_kp_feat
    
        self.lig_encoder = nn.Sequential(
            nn.Linear(atom_nf, 64),
            act_fn(),
            nn.Linear(64, hidden_nf),
            act_fn()
        )

        self.lig_decoder = nn.Sequential(
            nn.Linear(hidden_nf, 2 * atom_nf),
            act_fn(),
            nn.Linear(2 * atom_nf, atom_nf)
        )

        if rec_nf != hidden_nf:
            self.rec_encoder = nn.Sequential(
                nn.Linear(rec_nf, 2 * rec_nf),
                act_fn(),
                nn.Linear(2 * rec_nf, hidden_nf),
                act_fn()
            )
        else:
            self.rec_encoder = nn.Identity()

        # we add +1 to the feature size for the timestep
        self.egnn = LigRecEGNN(n_layers=n_layers, in_size=hidden_nf+1, hidden_size=hidden_nf+1, 
                               out_size=hidden_nf+1, use_tanh=use_tanh, 
                               message_norm=message_norm, update_kp_feat=update_kp_feat, norm=norm)


    def forward(self, g: dgl.DGLHeteroGraph, timestep, lig_batch_idx, kp_batch_idx):
        # inputs: ligand/keypoint positions/features, timestep
        # outputs: predicted noise

        with g.local_scope():

            # get initial lig and rec features from graph
            lig_feat = g.nodes['lig'].data['h_0']
            kp_feat = g.nodes['kp'].data['h_0']

            # encode lig/rec features
            lig_feat = self.lig_encoder(lig_feat)
            kp_feat = self.rec_encoder(kp_feat)

            # add timestep to node features
            t_lig = timestep[lig_batch_idx].view(-1, 1)
            lig_feat = torch.concatenate([lig_feat, t_lig], dim=1)

            t_kp = timestep[kp_batch_idx].view(-1, 1)
            kp_feat = torch.concatenate([kp_feat, t_kp], dim=1)

            # set lig and rec encoded features in the graph
            g.nodes['lig'].data['h_0'] = lig_feat
            g.nodes['kp'].data['h_0'] = kp_feat

            # add lig-lig and kp<->lig edges to graph
            g = self.add_lig_edges(g, lig_batch_idx, kp_batch_idx)

            # pass through convolutions and get updated h and x for the ligand
            h, x = self.egnn(g, lig_batch_idx, kp_batch_idx)

            # slice off time dimension
            h = h[:, :-1]

            # decode lig features and substract off original node coordinates from predicted ones
            # these become our noise predictions, \hat{\epsilon}
            eps_h = self.lig_decoder(h) 
            eps_x = x - g.nodes["lig"].data["x_0"]

            self.remove_lig_edges(g)

            return eps_h, eps_x

    def add_lig_edges(self, g: dgl.DGLHeteroGraph, lig_batch_idx, kp_batch_idx) -> dgl.DGLHeteroGraph:

        batch_num_nodes, batch_num_edges = get_batch_info(g)
        batch_size = g.batch_size

        # add lig-lig edges
        ll_idxs = radius_graph(g.nodes['lig'].data['x_0'], r=self.graph_cutoffs['ll'], batch=lig_batch_idx, max_num_neighbors=200)
        g.add_edges(ll_idxs[0], ll_idxs[1], etype='ll')

        # compute kp -> lig edges
        kl_idxs = radius(x=g.nodes['lig'].data['x_0'], y=g.nodes['kp'].data['x'], batch_x=lig_batch_idx, batch_y=kp_batch_idx, r=self.graph_cutoffs['kl'], max_num_neighbors=100)
        g.add_edges(kl_idxs[0], kl_idxs[1], etype='kl')

        # compute batch information
        batch_num_edges[('lig', 'll', 'lig')] = get_edges_per_batch(ll_idxs[0], batch_size, lig_batch_idx)
        kl_edges_per_batch = get_edges_per_batch(kl_idxs[0], batch_size, kp_batch_idx)
        batch_num_edges[('kp', 'kl', 'lig')] = kl_edges_per_batch

        # add lig -> kp edges if necessary
        if self.update_kp_feat:
            g.add_edges(kl_idxs[1], kl_idxs[0], etype='lk')
            batch_num_edges[('lig', 'lk', 'kp')] = kl_edges_per_batch

        # update the graph's batch information
        g.set_batch_num_edges(batch_num_edges)
        g.set_batch_num_nodes(batch_num_nodes)

        return g
    
    def remove_lig_edges(self, g: dgl.DGLHeteroGraph):

        if self.update_kp_feat:
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