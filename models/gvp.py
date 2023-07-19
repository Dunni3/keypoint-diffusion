import torch
from torch import nn, einsum
import dgl
import dgl.function as fn
from typing import List, Tuple, Union, Dict

# helper functions
def exists(val):
    return val is not None

# the classes GVP, GVPDropout, and GVPLayerNorm are taken from lucidrains' geometric-vector-perceptron repository
# https://github.com/lucidrains/geometric-vector-perceptron/tree/main

class GVP(nn.Module):
    def __init__(
        self,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        hidden_vectors = None,
        feats_activation = nn.Sigmoid(),
        vectors_activation = nn.Sigmoid(),
        vector_gating = False
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in

        self.dim_vectors_out = dim_vectors_out
        dim_h = max(dim_vectors_in, dim_vectors_out) if hidden_vectors is None else hidden_vectors

        self.Wh = nn.Parameter(torch.randn(dim_vectors_in, dim_h))
        self.Wu = nn.Parameter(torch.randn(dim_h, dim_vectors_out))

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + dim_feats_in, dim_feats_out),
            feats_activation
        )

        # branching logic to use old GVP, or GVP with vector gating

        self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out) if vector_gating else None

    def forward(self, data):
        feats, vectors = data
        b, n, _, v, c  = *feats.shape, *vectors.shape

        assert c == 3 and v == self.dim_vectors_in, 'vectors have wrong dimensions'
        assert n == self.dim_feats_in, 'scalar features have wrong dimensions'

        Vh = einsum('b v c, v h -> b h c', vectors, self.Wh)
        Vu = einsum('b h c, h u -> b u c', Vh, self.Wu)

        sh = torch.norm(Vh, p = 2, dim = -1)

        s = torch.cat((feats, sh), dim = 1)

        feats_out = self.to_feats_out(s)

        if exists(self.scalar_to_vector_gates):
            gating = self.scalar_to_vector_gates(feats_out)
            gating = gating.unsqueeze(dim = -1)
        else:
            gating = torch.norm(Vu, p = 2, dim = -1, keepdim = True)

        vectors_out = self.vectors_activation(gating) * Vu
        return (feats_out, vectors_out)
    
class GVPDropout(nn.Module):
    """ Separate dropout for scalars and vectors. """
    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = nn.Dropout2d(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)

class GVPLayerNorm(nn.Module):
    """ Normal layer norm for scalars, nontrainable norm for vectors. """
    def __init__(self, feats_h_size, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, feats, vectors):
        vector_norm = vectors.norm(dim=(-1,-2), keepdim=True)
        normed_feats = self.feat_norm(feats)
        normed_vectors = vectors / (vector_norm + self.eps)
        return normed_feats, normed_vectors
    


class GVPEdgeConv(nn.Module):

    """GVP graph convolution on a single edge type on a heterogenous graph."""

    def __init__(self, edge_type: Tuple[str, str, str], scalar_size: int = 128, vector_size: int = 16,
                  scalar_activation=nn.SiLU, vector_activation=nn.Sigmoid,
                  n_message_gvps: int = 1, n_update_gvps: int = 1,
                  use_dst_feats: bool = True,
                  edge_feat_size=int, coords_range=10, message_norm: Union[float, str] = 10, dropout: float = 0.0):
        
        super().__init__()

        self.edge_type = edge_type
        self.src_ntype = edge_type[0]
        self.dst_ntype = edge_type[2]
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.edge_feat_size = edge_feat_size
        self.use_dst_feats = use_dst_feats

        # create message passing function
        message_gvps = []
        for i in range(n_message_gvps):

            dim_vectors_in = vector_size
            dim_feats_in = scalar_size

            # on the first layer, there is an extra edge vector for the displacement vector between the two node positions
            if i == 0:
                dim_vectors_in += 1
                
            # if this is the first layer and we are using destination node features to compute messages, add them to the input dimensions
            if use_dst_feats and i == 0:
                dim_vectors_in += vector_size
                dim_feats_in += scalar_size

            message_gvps.append(
                GVP(dim_vectors_in=dim_vectors_in, 
                    dim_vectors_out=vector_size, 
                    dim_feats_in=dim_feats_in, 
                    dim_feats_out=scalar_size, 
                    feats_activation=scalar_activation(), 
                    vectors_activation=vector_activation(), 
                    vector_gating=True)
            )
        self.edge_message = nn.Sequential(*message_gvps)

        # create update function
        update_gvps = []
        for i in range(n_update_gvps):
            update_gvps.append(
                GVP(dim_vectors_in=vector_size, 
                    dim_vectors_out=vector_size, 
                    dim_feats_in=scalar_size, 
                    dim_feats_out=scalar_size, 
                    feats_activation=scalar_activation(), 
                    vectors_activation=vector_activation(), 
                    vector_gating=True)
            )
        self.node_update = nn.Sequential(*update_gvps)
        
        self.dropout = GVPDropout(self.dropout_rate)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)
        self.update_layer_norm = GVPLayerNorm(self.scalar_size)

        if self.message_norm == 'mean':
            self.agg_func = fn.mean
        else:
            self.agg_func = fn.sum


    def forward(self, g: dgl.DGLHeteroGraph, 
                src_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                edge_feats: torch.Tensor = None, 
                dst_feats: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None] = None,
                z: Union[float, torch.Tensor] = 1):
        # vec_feat has shape (n_nodes, n_vectors, 3)

        with g.local_scope():


            # set node features
            if self.src_ntype == self.dst_ntype:
                scalar_feat, coord_feat, vec_feat = src_feats
                g.nodes[self.src_ntype].data["h"] = scalar_feat
                g.nodes[self.src_ntype].data["x"] = coord_feat
                g.nodes[self.src_ntype].data["v"] = vec_feat
            else:
                scalar_feat, coord_feat, vec_feat = src_feats
                g.nodes[self.src_ntype].data["h"] = scalar_feat
                g.nodes[self.src_ntype].data["x"] = coord_feat
                g.nodes[self.src_ntype].data["v"] = vec_feat

                scalar_feat, coord_feat, vec_feat = dst_feats
                g.nodes[self.dst_ntype].data["h"] = scalar_feat
                g.nodes[self.dst_ntype].data["x"] = coord_feat
                g.nodes[self.dst_ntype].data["v"] = vec_feat

            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feats is not None, "Edge features must be provided."
                g.edges[self.edge_type].data["a"] = edge_feats

            # get vectors between node positions
            g.apply_edges(fn.u_sub_v("x", "x", "x_diff"), etype=self.edge_type)

            # normalize x_diff
            # i don't think this is necessary but i'm leaving it here for now
            # g.edges['rr'].data['x_diff'] = g.edges['rr'].data['x_diff'] / (torch.norm(g.edges['rr'].data['x_diff'], dim=-1, keepdim=True) + 1e-10 )

            # compute messages on every edge
            g.apply_edges(self.message, etype=self.edge_type)

            # aggregate messages from every edge
            g.update_all(fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg"), etype=self.edge_type)
            g.update_all(fn.copy_e("vec_msg", "m"), self.agg_func("m", "vec_msg"), etype=self.edge_type)

            # get aggregated scalar and vector messages
            scalar_msg = g.nodes[self.dst_ntype].data["scalar_msg"] / z
            vec_msg = g.nodes[self.dst_ntype].data["vec_msg"] / z

            # dropout scalar and vector messages
            scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)

            # update scalar and vector features, apply layernorm
            scalar_feat = g.nodes[self.dst_ntype].data['h'] + scalar_msg
            vec_feat = g.nodes[self.dst_ntype].data['v'] + vec_msg
            scalar_feat, vec_feat = self.message_layer_norm(scalar_feat, vec_feat)

            # apply node update function, apply dropout to residuals, apply layernorm
            scalar_residual, vec_residual = self.node_update((scalar_feat, vec_feat))
            scalar_residual, vec_residual = self.dropout(scalar_residual, vec_residual)
            scalar_feat = scalar_feat + scalar_residual
            vec_feat = vec_feat + vec_residual
            scalar_feat, vec_feat = self.update_layer_norm(scalar_feat, vec_feat)

        return scalar_feat, vec_feat

    def message(self, edges):

        # concatenate x_diff and v on every edge to produce vector features
        if self.use_dst_feats:
            vec_feats = torch.cat([edges.data["x_diff"].unsqueeze(1), edges.src["v"], edges.dst["v"]], dim=1)
        else:
            vec_feats = torch.cat([edges.data["x_diff"].unsqueeze(1), edges.src["v"]], dim=1)

        # create scalar features
        if self.edge_feat_size > 0 and self.use_dst_feats:
            scalar_feats = torch.cat([edges.src["h"], edges.dst["h"], edges.data['a']], dim=1)
        elif self.edge_feat_size > 0:
            scalar_feats = torch.cat([edges.src["h"], edges.data['a'] ], dim=1)
        else:
            scalar_feats = edges.src["h"]

        scalar_message, vector_message = self.edge_message((scalar_feats, vec_feats))

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}

class GVPMultiEdgeConv(nn.Module):

    """GVP graph convolution over multiple edge types for a heterogeneous graph."""

    def __init__(self, etypes: List[Tuple[str, str, str]], scalar_size: int = 128, vector_size: int = 16,
                  scalar_activation=nn.SiLU, vector_activation=nn.Sigmoid,
                  n_message_gvps: int = 1, n_update_gvps: int = 1, coords_range=10, 
                  message_norm: Union[float, str, Dict] = 10, dropout: float = 0.0):
        
        super().__init__()

        self.etypes = etypes
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.droupout_rate = dropout

        # get all node types that are the destination of an edge type
        self.dst_ntypes = set([edge_type[2] for edge_type in self.etypes])
            
        # check that message_norm is valid
        self.check_message_norm(message_norm)

        # we need to construct a dictionary of normalization values for each destination node type
        self.norm_values = {}
        for ntype in self.dst_ntypes:
            if isinstance(message_norm, dict):
                nv = message_norm[ntype]
            else:
                nv = message_norm

            if nv == 'mean':
                self.norm_values[ntype] = 1.0
            else:
                self.norm_values[ntype] = nv

        # set the aggregation function
        if isinstance(message_norm, (dict, int, float)):
            self.agg_func = fn.sum
        elif message_norm == 'mean':
            self.agg_func = fn.mean

        # create message functions for each edge type
        self.edge_message_fns = nn.ModuleDict()
        for etype in self.etypes:
            edge_message_gvps = []
            for i in range(n_message_gvps):

                if i == 0:
                    dim_vectors_in = vector_size + 1
                else:
                    dim_vectors_in = vector_size

                edge_message_gvps.append(
                    GVP(dim_vectors_in=dim_vectors_in, 
                        dim_vectors_out=vector_size, 
                        dim_feats_in=scalar_size, 
                        dim_feats_out=scalar_size, 
                        feats_activation=scalar_activation(), 
                        vectors_activation=vector_activation(), 
                        vector_gating=True)
                )

            self.edge_message_fns[etype] = nn.Sequential(*edge_message_gvps)

        # create node update functions for each node type
        self.node_update_fns = nn.ModuleDict()
        self.update_layer_norms = nn.ModuleDict()
        self.message_layer_norms = nn.ModuleDict()
        for ntype in self.dst_ntypes:
            update_gvps = []
            for i in range(n_update_gvps):
                update_gvps.append(
                    GVP(dim_vectors_in=vector_size, 
                        dim_vectors_out=vector_size, 
                        dim_feats_in=scalar_size, 
                        dim_feats_out=scalar_size, 
                        feats_activation=scalar_activation(), 
                        vectors_activation=vector_activation(), 
                        vector_gating=True)
                )
            self.node_update_fns[ntype] = nn.Sequential(*update_gvps)
            self.message_layer_norms[ntype] = GVPLayerNorm(scalar_size)
            self.update_layer_norms[ntype] = GVPLayerNorm(scalar_size)

        self.dropout = GVPDropout(self.dropout_rate)

    def check_message_norm(self, message_norm: Union[float, int, str, dict]):
            
        invalid_message_norm = False

        if isinstance(message_norm, str) and message_norm != 'mean':
            invalid_message_norm = True
        elif isinstance(message_norm, (float, int)):
            if message_norm <= 0:
                invalid_message_norm = True
        elif isinstance(message_norm, dict):
            if not all( isinstance(val, (float, int)) for val in message_norm.values() ):
                invalid_message_norm = True
            if not all( val > 0 for val in message_norm.values() ):
                invalid_message_norm = True
            if not all( key in message_norm for key in self.dst_ntypes.keys() ):
                raise ValueError(f"message_norm dictionary must contain keys for all destination node types. got keys for {message_norm.keys()} but needed keys for {self.dst_ntypes.keys()}")
        
        if invalid_message_norm:
            raise ValueError(f"message_norm values must be 'mean' or a positive number, got {message_norm}")


    def forward(self, g: dgl.DGLHeteroGraph, node_feats: Dict[str, Tuple]):
        

        with g.local_scope():

            # set node features
            for ntype in node_feats:
                scalar_feats, pos_feats, vec_feats = node_feats[ntype]
                g.nodes[ntype].data['h'] = scalar_feats
                g.nodes[ntype].data['v'] = vec_feats
                g.nodes[ntype].data['x'] = pos_feats

            # compute edge messages
            for etype in self.etypes:
                g.apply_edges(self.message, etype=etype)

            # aggregate scalar messages
            update_dict = {}
            for etype in self.etypes:
                update_dict[etype] = (fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg"),)
            g.multi_update_all(update_dict, cross_reducer="sum")

            # aggregate vector messages
            update_dict = {}
            for etype in self.etypes:
                update_dict[etype] = (fn.copy_e("vector_msg", "m"), self.agg_func("m", "vector_msg"),)
            g.multi_update_all(update_dict, cross_reducer="sum")

            # apply dropout, layernorm, and add to original features
            output_feats = {}
            for ntype in self.dst_ntypes:
                scalar_feats, vec_feats = g.nodes[ntype].data["h"], g.nodes[ntype].data["v"]
                scalar_msg = g.nodes[ntype].data["scalar_msg"] / self.norm_values[ntype]
                vec_msg = g.nodes[ntype].data["vector_msg"] / self.norm_values[ntype]
                scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)
                scalar_feats += scalar_msg
                vec_feats += vec_msg
                scalar_feats, vec_feats = self.message_layer_norms[ntype](scalar_feats, vec_feats)

                # apply node update function
                scalar_res, vec_res = self.node_update_fns[ntype]((scalar_feats, vec_feats))
                scalar_res, vec_res = self.dropout(scalar_res, vec_res)
                scalar_feats += scalar_res
                vec_feats += vec_res
                scalar_feats, vec_feats = self.update_layer_norms[ntype](scalar_feats, vec_feats)

                pos_feats = g.nodes[ntype].data["x"]

                output_feats[ntype] = (scalar_feats, pos_feats, vec_feats)

        return output_feats    

    def message(self, edges):

        edge_type = edges.canonical_etype

        vec_feats = torch.cat([edges.data["x_diff"].unsqueeze(1), edges.src["v"]], dim=1)

        scalar_feats = edges.src["h"]

        scalar_message, vector_message = self.edge_message_fns[edge_type]((scalar_feats, vec_feats))

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}