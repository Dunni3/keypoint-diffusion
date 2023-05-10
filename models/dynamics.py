import torch
import torch.nn as nn
from typing import Dict, List
import dgl.function as fn
import dgl

class LigRecConv(nn.Module):

    # this class was adapted from the DGL implementation of EGNN
    # original code: https://github.com/dmlc/dgl/blob/76bb54044eb387e9e3009bc169e93d66aa004a74/python/dgl/nn/pytorch/conv/egnnconv.py
    # I have extended the EGNN graph conv layer to operate on heterogeneous graphs containing containing receptor and ligand nodes

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, use_tanh=False, coords_range=10, message_norm=1):
        super().__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()
        self.use_tanh = use_tanh
        self.message_norm = message_norm

        self.coords_range = coords_range

        self.edge_types = ["ll", "rl"]

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

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

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
        if self.use_tanh:
            msg_x = torch.tanh( self.coord_mlp[edge_type](f) )* edges.data["x_diff"] * self.coords_range
        else:
            msg_x = self.coord_mlp[edge_type](f)*edges.data["x_diff"]

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
                graph.apply_edges(self.compute_dij, etype=etype)

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
            h_neigh, x_neigh = graph.ndata["h_neigh"]/self.message_norm, graph.ndata["x_neigh"]/self.message_norm

            # compute updated features/coordinates
            # note that the receptor features/coordinates are not updated
            node_mlp_input = torch.cat([node_feat['lig'], h_neigh['lig']], dim=-1)
            h = {'lig': node_feat['lig'] + self.node_mlp(node_mlp_input),
                 'rec': node_feat['rec']}

            x = {'lig': coord_feat['lig'] + x_neigh['lig'],
                 'rec': coord_feat['rec']}

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

    def __init__(self, n_layers, in_size, hidden_size, out_size, use_tanh=False, message_norm=1):
        super().__init__()

        self.n_layers = n_layers
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

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
                LigRecConv(in_size=layer_in_size, hidden_size=layer_hidden_size, out_size=layer_out_size, use_tanh=use_tanh, message_norm=message_norm)
            )

            self.conv_layers = nn.ModuleList(self.conv_layers)

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

    def __init__(self, atom_nf, rec_nf, n_layers=4, hidden_nf=255, act_fn=nn.SiLU, receptor_keypoint_k=6, ligand_k=8, use_tanh=False, message_norm=1):
        super().__init__()

        self.receptor_keypoint_k = receptor_keypoint_k
        self.ligand_k = ligand_k    

    
        self.lig_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn(),
            nn.Linear(2 * atom_nf, hidden_nf)
        )

        self.lig_decoder = nn.Sequential(
            nn.Linear(hidden_nf, 2 * atom_nf),
            act_fn(),
            nn.Linear(2 * atom_nf, atom_nf)
        )

        self.rec_encoder = nn.Sequential(
            nn.Linear(rec_nf, 2 * rec_nf),
            act_fn(),
            nn.Linear(2 * rec_nf, hidden_nf)
        )

        # we add +1 to the feature size for the timestep
        self.egnn = LigRecEGNN(n_layers=n_layers, in_size=hidden_nf+1, hidden_size=hidden_nf+1, out_size=hidden_nf+1, use_tanh=use_tanh, message_norm=message_norm)


    def forward(self, lig_pos, lig_feat, rec_pos, rec_feat, timestep, unbatch_eps=False):
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
            lig_feat_time.append( torch.cat([lig_feat[i], t_reshaped], dim=1) )

            t_reshaped = timestep[i].repeat(rec_feat[i].shape[0]).view(-1,1)
            rec_feat_time.append( torch.cat([rec_feat[i], t_reshaped], dim=1) )

        # construct heterograph
        batched_graph = self.make_graph(lig_pos, lig_feat_time, rec_pos, rec_feat_time)

        # check that graph batching simply concats features - it does!
        # this means we can return the batched eps predictions and then the ligand diffusion model
        # can simply concat the added noise for input to the loss function

        # pass through LigRecEGNN
        h, x = self.egnn(batched_graph)
        
        # slice off time dimension
        h_final = h['lig'][:, :-1]

        # decode lig features and substract off original node coordinates from predicted ones
        # these become our noise predictions, \hat{\epsilon}
        eps_h = self.lig_decoder(h_final) 
        eps_x = x['lig'] - batched_graph.nodes["lig"].data["x_0"]

        # unbatch noise estimates - this is necessary when sampling because
        # we are going to use these noise estimates to denoise the input data.
        # during training, we just compute the l2 loss over all the predicted noise values
        # so we can deal with all of the noise predictions being lumped into one tensor
        if unbatch_eps:
            batched_graph.nodes["lig"].data["eps_h"] = eps_h
            batched_graph.nodes["lig"].data["eps_x"] = eps_x

            unbatched_graphs = dgl.unbatch(batched_graph)
            eps_h = [ g.nodes['lig'].data['eps_h'] for g in unbatched_graphs ]
            eps_x = [ g.nodes['lig'].data['eps_x'] for g in unbatched_graphs ]
 
        return eps_h, eps_x


    def make_graph(self, lig_pos, lig_feat, rec_pos, rec_feat):

        # note that all arguments except timestep are tuples of length batch_size containing
        # the values for each datapoint in the batch

        device = lig_pos[0].device
        
        graphs = []
        for i in range(len(lig_pos)):

            # create graph containing just ligand-ligand edges
            # TODO: expose all these k's as hyperparameters. Also, there is an issue when the ligand has less than k atoms. Maybe fix this by some method other than excluding small ligands?
            lig_graph = dgl.knn_graph(lig_pos[i], k=self.ligand_k, algorithm="bruteforce-blas", dist='euclidean', exclude_self=True).to(device)

            # find edges for rec -> lig conections
            rl_dist = torch.cdist(rec_pos[i], lig_pos[i]) # distance between every receptor keypoint and every ligand atom
            topk_dist, topk_idx = torch.topk(rl_dist, k=self.receptor_keypoint_k, dim=1, largest=False) # get k closest ligand atoms to each receptor key point

            # get list of rec -> ligand edges
            n_rec_nodes = rec_pos[i].shape[0]
            src_nodes = torch.repeat_interleave(torch.arange(n_rec_nodes), repeats=self.receptor_keypoint_k).to(device)
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