from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import dgl.function as dglfn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch_scatter import segment_coo, segment_csr

from losses.rec_encoder_loss import ReceptorEncoderLoss
from losses.dist_hinge_loss import DistanceHingeLoss
from models.dynamics import LigRecDynamics
from models.receptor_encoder import ReceptorEncoder
from models.n_nodes_dist import LigandSizeDistribution
from utils import get_batch_info, get_nodes_per_batch, copy_graph
from torch_scatter import segment_csr

class LigandDiffuser(nn.Module):

    def __init__(self, atom_nf, rec_nf, processed_dataset_dir: Path, n_timesteps: int = 1000, keypoint_centered=False, graph_config={},
    dynamics_config = {}, rec_encoder_config = {}, rec_encoder_loss_config= {}, precision=1e-4, lig_feat_norm_constant=1, rl_dist_threshold=0, use_fake_atoms=False):
        super().__init__()

        # NOTE: keypoint_centered is deprecated. This flag no longer has any effect. It is kept as an argument for backwards compatibility with previously trained models.

        self.n_lig_features = atom_nf
        self.n_kp_feat = rec_nf
        self.n_timesteps = n_timesteps
        self.lig_feat_norm_constant = lig_feat_norm_constant
        self.use_fake_atoms = use_fake_atoms
        
        # create the receptor -> ligand hinge loss if called for
        if rl_dist_threshold > 0:
            self.apply_rl_hinge_loss = True
            self.rl_hinge_loss_fn = DistanceHingeLoss(distance_threshold=rl_dist_threshold)
        else:
            self.apply_rl_hinge_loss = False

        # create ligand node distribution for sampling
        self.lig_size_dist = LigandSizeDistribution(processed_dataset_dir=processed_dataset_dir)

        # create noise schedule and dynamics model
        self.gamma = PredefinedNoiseSchedule(noise_schedule='polynomial_2', timesteps=n_timesteps, precision=precision)

        if 'no_cg' in rec_encoder_config:        
            self.dynamics = LigRecDynamics(atom_nf, rec_nf, no_cg=rec_encoder_config['no_cg'], **graph_config, **dynamics_config)
        else:
            self.dynamics = LigRecDynamics(atom_nf, rec_nf, **graph_config, **dynamics_config)

        # create receptor encoder and its loss function
        self.rec_encoder = ReceptorEncoder(**graph_config, **rec_encoder_config)
        self.rec_encoder_loss_fn = ReceptorEncoderLoss(**rec_encoder_loss_config)

    def forward(self, complex_graphs: dgl.DGLHeteroGraph, interface_points: List[torch.Tensor]):
        """Computes loss."""
        
        losses = {}

        # compute index pointers for segmented operations
        # batch_size = complex_graphs.batch_size
        # node_segidxs = {}
        # for ntype in complex_graphs.ntypes:
        #     segidx = torch.zeros(batch_size +1 , dtype=torch.int32, device=complex_graphs.device)
        #     segidx[1:] = complex_graphs.batch_num_nodes(ntype=ntype)
        #     node_segidxs[ntype] = torch.cumsum(segidx, 0)

        # edge_segidxs = {}
        # for etype in complex_graphs.etypes:
        #     segidx = torch.zeros(batch_size +1 , dtype=torch.int64, device=complex_graphs.device)
        #     segidx[1:] = complex_graphs.batch_num_edges(etype=etype)
        #     edge_segidxs[etype] = torch.cumsum(segidx, 0)

        # normalize values
        complex_graphs = self.normalize(complex_graphs)

        batch_size = complex_graphs.batch_size
        device = complex_graphs.device

        # get batch indicies of every ligand and keypoint - useful later
        batch_idx = torch.arange(batch_size, device=device)
        lig_batch_idx = batch_idx.repeat_interleave(complex_graphs.batch_num_nodes('lig'))
        kp_batch_idx = batch_idx.repeat_interleave(complex_graphs.batch_num_nodes('kp'))
                
        # encode the receptor
        complex_graphs = self.rec_encoder(complex_graphs, kp_batch_idx)

        # if we are applying the RL hinge loss, we will need to be able to put receptor atoms and the ligand into the same
        # referance frame. in order to do this, we need the initial COM of the keypoints
        if self.apply_rl_hinge_loss:
            init_kp_com = dgl.readout_nodes(complex_graphs, feat='x', ntype='kp', op='mean')

        # compute receptor encoding loss
        losses['rec_encoder'] = self.rec_encoder_loss_fn(complex_graphs, interface_points=interface_points)

        # remove ligand COM from receptor/ligand complex
        complex_graphs = self.remove_com(complex_graphs, lig_batch_idx, kp_batch_idx, com='ligand')

        # sample timepoints for each item in the batch
        t = torch.randint(0, self.n_timesteps, size=(batch_size,), device=device).float() # timesteps
        t = t / self.n_timesteps

        # sample epsilon for each ligand
        eps = {
            'h':torch.randn(complex_graphs.nodes['lig'].data['h_0'].shape, device=device),
            'x':torch.randn(complex_graphs.nodes['lig'].data['x_0'].shape, device=device)
        }
        
        # construct noisy versions of the ligand
        gamma_t = self.gamma(t).to(device=device)
        complex_graphs = self.noised_representation(complex_graphs, lig_batch_idx, kp_batch_idx, eps, gamma_t)

        # predict the noise that was added
        eps_h_pred, eps_x_pred = self.dynamics(complex_graphs, t, lig_batch_idx, kp_batch_idx)

        # compute hinge loss if necessary
        if self.apply_rl_hinge_loss:

            with complex_graphs.local_scope():

                # predict denoised ligand
                g_denoised = self.denoised_representation(complex_graphs, lig_batch_idx, kp_batch_idx, eps_x_pred, eps_h_pred, gamma_t)

                # translate ligand back to intitial frame of reference
                g_denoised = self.remove_com(g_denoised, lig_batch_idx, kp_batch_idx, com='receptor')
                g_denoised.nodes['lig'].data['x_0'] = g_denoised.nodes['lig'].data['x_0'] + init_kp_com[lig_batch_idx]

                # compute hinge loss between ligand atom position and receptor atom positions
                rl_hinge_loss = 0
                for g in dgl.unbatch(g_denoised):
                    denoised_lig_pos = g.nodes['lig'].data['x_0']
                    rec_atom_pos = g.nodes['rec'].data['x_0']
                    rl_hinge_loss += self.rl_hinge_loss_fn(denoised_lig_pos, rec_atom_pos)

                losses['rl_hinge'] = rl_hinge_loss

        # compute l2 loss on noise
        if self.use_fake_atoms:
            # real_atom_mask = torch.concat([ ~(lig_feat[:, -1].bool()) for lig_feat in lig_atom_features ])[:, None]
            real_atom_mask = complex_graphs.nodes['lig'].data['h_0'][:, -1:].bool()
            n_real_atoms = real_atom_mask.sum()
            n_x_loss_terms = n_real_atoms*3
            x_loss = ((eps['x'] - eps_x_pred)*real_atom_mask).square().sum() # mask out loss on predicted position of fake atoms
        else:
            x_loss = ((eps['x'] - eps_x_pred)).square().sum()
            n_x_loss_terms = eps['x'].numel()

        h_loss = (eps['h'] - eps_h_pred).square().sum()
        losses['l2'] = (x_loss + h_loss) / (n_x_loss_terms + eps['h'].numel())

        losses['pos'] = x_loss / n_x_loss_terms
        losses['feat'] = h_loss / eps['h'].numel()

        return losses
    
    def normalize(self, complex_graphs: dgl.DGLHeteroGraph):
        complex_graphs.nodes['lig'].data['h_0'] = complex_graphs.nodes['lig'].data['h_0'] / self.lig_feat_norm_constant
        return complex_graphs

    def unnormalize(self, complex_graphs: dgl.DGLHeteroGraph):
        complex_graphs.nodes['lig'].data['h_0'] = complex_graphs.nodes['lig'].data['h_0'] * self.lig_feat_norm_constant
        return complex_graphs

    def remove_com(self, complex_graphs, lig_batch_idx, rec_batch_idx, com: str = None):
        """Remove center of mass from ligand atom positions and receptor keypoint positions.

        This method can remove either the ligand COM, receptor keypoint COM or the complex COM.
        """               
        if com is None:
            raise NotImplementedError('removing COM of receptor/ligand complex not implemented')
        elif com == 'ligand':
            ntype = 'lig'
            feat = 'x_0'
        elif com == 'receptor':
            ntype = 'kp'
            feat = 'x'
        else:
            raise ValueError(f'invalid value for com: {com=}')
        
        com = dgl.readout_nodes(complex_graphs, feat=feat, ntype=ntype, op='mean')

        complex_graphs.nodes['lig'].data['x_0'] = complex_graphs.nodes['lig'].data['x_0'] - com[lig_batch_idx]
        complex_graphs.nodes['kp'].data['x'] = complex_graphs.nodes['kp'].data['x_0'] - com[rec_batch_idx]
        return complex_graphs

    def noised_representation(self, g: dgl.DGLHeteroGraph, lig_batch_idx: torch.Tensor, kp_batch_idx: torch.Tensor,
                              eps: Dict[str, torch.Tensor], gamma_t: torch.Tensor):
        

        alpha_t = self.alpha(gamma_t)[lig_batch_idx][:, None]
        sigma_t = self.sigma(gamma_t)[lig_batch_idx][:, None]

        g.nodes['lig'].data['x_0'] = alpha_t*g.nodes['lig'].data['x_0'] + sigma_t*eps['x']
        g.nodes['lig'].data['h_0'] = alpha_t*g.nodes['lig'].data['h_0'] + sigma_t*eps['h']
        

        # remove ligand COM from the system
        g = self.remove_com(g, lig_batch_idx, kp_batch_idx, com='ligand')
        
        return g
    
    def denoised_representation(self, g: dgl.DGLHeteroGraph, lig_batch_idx: torch.Tensor, kp_batch_idx: torch.Tensor,
                              eps_x_pred: torch.Tensor, eps_h_pred: torch.Tensor, gamma_t: torch.Tensor):
        # assuming the input ligand COM is zero, we compute the denoised verison of the ligand
        alpha_t = self.alpha(gamma_t)[lig_batch_idx][:, None]
        sigma_t = self.sigma(gamma_t)[lig_batch_idx][:, None]

        g.nodes['lig'].data['x_0'] = (g.nodes['lig'].data['x_0'] - sigma_t*eps_x_pred)/alpha_t
        g.nodes['lig'].data['h_0'] = (g.nodes['lig'].data['h_0'] - sigma_t*eps_h_pred)/alpha_t

        return g
    
    def sigma(self, gamma):
        """Computes sigma given gamma."""
        return torch.sqrt(torch.sigmoid(gamma))

    def alpha(self, gamma):
        """Computes alpha given gamma."""
        return torch.sqrt(torch.sigmoid(-gamma))

    def sigma_and_alpha_t_given_s(self, gamma_t, gamma_s):
        # this function is almost entirely copied from DiffSBDD

        sigma2_t_given_s = -torch.expm1(fn.softplus(gamma_s) - fn.softplus(gamma_t))

        log_alpha2_t = fn.logsigmoid(-gamma_t)
        log_alpha2_s = fn.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def encode_receptors(self, g: dgl.DGLHeteroGraph):
        # this function is used to encode receptors ONLY during sampling/

        device = g.device

        # compute initial receptor atom COM
        init_rec_atom_com = dgl.readout_nodes(g, feat='x_0', op='mean', ntype='rec')

        # get batch indicies of every ligand and keypoint - useful later
        batch_idx = torch.arange(g.batch_size, device=device)
        kp_batch_idx = batch_idx.repeat_interleave(g.batch_num_nodes('kp'))

        # get keypoints positions/features
        g = self.rec_encoder(g, kp_batch_idx)

        # get initial keypoint center of mass
        init_kp_com = dgl.readout_nodes(g, feat='x_0', op='mean', ntype='kp')

        # remove (receptor atom COM, or keypoint COM) from receptor keypoints
        # TODO: does this effect sampling performance? there is an argument to be made to starting sampling at keypoint COM?
        g.nodes['kp'].data['x_0'] = g.nodes['kp'].data['x_0'] - init_rec_atom_com[kp_batch_idx]

        return g, init_kp_com

    
    @torch.no_grad()
    def _sample(self, ref_graphs: List[dgl.DGLHeteroGraph], n_lig_atoms: List[List[int]], rec_enc_batch_size: int = 32, diff_batch_size: int = 32, visualize=False) -> List[List[Dict[str, torch.Tensor]]]:
        """Sample multiple receptors with multiple ligands per receptor.

        Args:
            receptors (List[dgl.DGLGraph]): A list containing a DGL graph of each receptor that is to be sampled.
            n_lig_atoms (List[List[int]]): A list that contains a list for each receptor. Each nested list contains integers that each specify the number of atoms in a ligand.
            rec_enc_batch_size (int, optional): Batch size for forward passes through receptor encoder. Defaults to 32.
            diff_batch_size (int, optional): Batch size for forward passes through denoising model. Defaults to 32.

        Returns:
            List[Dict[str, torch.Tensor]]: A list of length len(receptors). Each element of this list is a dictionary with keys "positions" and "features". The values are lists of tensors, one tensor per ligand. 
        """        

        device = ref_graphs[0].device
        n_receptors = len(ref_graphs)

        # encode all the receptors
        ref_graphs_batched = dgl.batch(ref_graphs)
        ref_graphs_batched, ref_init_kp_com = self.encode_receptors(ref_graphs_batched)

        # make copies of receptors and set the appropriate number of ligand atoms for all graphs
        graphs = []
        init_kp_coms = []
        for rec_idx, ref_graph in enumerate(dgl.unbatch(ref_graphs_batched)):
            n_lig_atoms_rec = n_lig_atoms[rec_idx]


            g_copies = copy_graph(ref_graph, n_copies=len(n_lig_atoms_rec), lig_atoms_per_copy=torch.tensor(n_lig_atoms_rec))
            kp_com_copies = [ ref_init_kp_com[rec_idx].clone() for _ in range(len(n_lig_atoms_rec)) ]

            graphs.extend(g_copies)
            init_kp_coms.extend(kp_com_copies)

        # proceed to batched sampling
        n_complexes = len(graphs)
        n_complexes_sampled = 0
        lig_pos, lig_feat = [], []
        for batch_idx in range(ceil(n_complexes / diff_batch_size)):

            # determine number of complexes that will be in this batch
            n_samples_batch = min(diff_batch_size, n_complexes - n_complexes_sampled)

            start_idx = batch_idx*diff_batch_size
            end_idx = start_idx + n_samples_batch

            batch_graphs = dgl.batch(graphs[start_idx:end_idx])
            batch_init_kp_com = torch.stack(init_kp_coms[start_idx:end_idx], dim=0)

            batch_lig_pos, batch_lig_feat = self.sample_from_encoded_receptors(batch_graphs, batch_init_kp_com, visualize=visualize)
            lig_pos.extend(batch_lig_pos)
            lig_feat.extend(batch_lig_feat)

            n_complexes_sampled += n_samples_batch

        # group sampled ligands by receptor
        samples = []
        end_idx = 0
        for rec_idx in range(n_receptors):
            n_ligands = len(n_lig_atoms[rec_idx])

            start_idx = end_idx
            end_idx = start_idx + n_ligands

            samples.append({
                'positions': lig_pos[start_idx:end_idx],
                'features': lig_feat[start_idx:end_idx]
            })

        return samples

    def sample_from_encoded_receptors(self, g: dgl.DGLHeteroGraph, init_kp_com: torch.Tensor, 
                                      visualize=False):

        device = g.device
        n_complexes = g.batch_size
        
        # get batch indicies of every ligand and keypoint - useful later
        batch_idx = torch.arange(g.batch_size, device=device)
        lig_batch_idx = batch_idx.repeat_interleave(g.batch_num_nodes('lig'))
        kp_batch_idx = batch_idx.repeat_interleave(g.batch_num_nodes('kp'))
        node_batch_idx_dict = {
            'lig': lig_batch_idx,
            'kp': kp_batch_idx
        }

        # sample initial positions/features of ligands
        for feat in ['x_0', 'h_0']:
            g.nodes['lig'].data[feat] = torch.randn(g.nodes['lig'].data[feat].shape, device=device)

        if visualize:
            
            # convert positions and features to cpu
            # convert positions to input frame of reference: remove current kp com and add original init kp com
            # note that this function assumes that the keypoints passed as arguments have the keypoint COM removed from them already, so all we need to do is add back in the initial keypoint COM

            # make a copy of g
            g_frame = copy_graph(g, n_copies=1, batched_graph=True)[0]

            # unnoramlize
            g_frame = self.unnormalize(g_frame)

            # move ligand back into initial frame of reference
            g_frame.nodes['lig'].data['x_0'] = g_frame.nodes['lig'].data['x_0'] + init_kp_com[lig_batch_idx]

            # remove fake atoms
            if self.use_fake_atoms:
                g_frame = self.remove_fake_atoms(g_frame, lig_batch_idx)
            
            lig_pos_frames = []
            lig_feat_frames = []
            g_frame = g_frame.to('cpu')
            lig_pos, lig_feat = [], []
            for g_i in dgl.unbatch(g_frame):
                lig_pos.append(g_i.nodes['lig'].data['x_0'])
                lig_feat.append(g_i.nodes['lig'].data['h_0'])
            lig_pos_frames.append(lig_pos)
            lig_feat_frames.append(lig_feat)
            


        # remove ligand com from every receptor/ligand complex
        g = self.remove_com(g, lig_batch_idx, kp_batch_idx, com='ligand')

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.n_timesteps)):
            s_arr = torch.full(size=(n_complexes,), fill_value=s, device=device)
            t_arr = s_arr + 1
            s_arr = s_arr / self.n_timesteps
            t_arr = t_arr / self.n_timesteps

            g = self.sample_p_zs_given_zt(s_arr, t_arr, g, lig_batch_idx, kp_batch_idx)
            # if g.batch_num_edges('ll').shape[0] != g.batch_size:
            #     print('problem!')
            if visualize:

                # make a copy of g
                g_frame = copy_graph(g, n_copies=1, batched_graph=True)[0]

                # unnoramlize
                g_frame = self.unnormalize(g_frame)

                # move ligand back into initial frame of reference
                kp_com = dgl.readout_nodes(g_frame, feat='x_0', ntype='kp', op='mean')
                delta = init_kp_com - kp_com
                g_frame.nodes['lig'].data['x_0'] = g_frame.nodes['lig'].data['x_0'] + delta[lig_batch_idx]

                # remove fake atoms
                if self.use_fake_atoms:
                    g_frame = self.remove_fake_atoms(g_frame, lig_batch_idx)

                # convert graph to cpu and split out ligand positions and features
                g_frame = g_frame.to('cpu')
                lig_pos, lig_feat = [], []
                for g_i in dgl.unbatch(g_frame):
                    lig_pos.append(g_i.nodes['lig'].data['x_0'])
                    lig_feat.append(g_i.nodes['lig'].data['h_0'])
                lig_pos_frames.append(lig_pos)
                lig_feat_frames.append(lig_feat)

        # remove keypoint COM from system after generation
        g = self.remove_com(g, lig_batch_idx, kp_batch_idx, com='receptor')

        # TODO: model P(x0 | x1)?

        # add initial keypoint COM to system, bringing us back into the input frame of reference
        for ntype in ['lig', 'kp']:
            g.nodes[ntype].data['x_0'] = g.nodes[ntype].data['x_0'] + init_kp_com[ node_batch_idx_dict[ntype] ]
            
        # unnormalize features
        g = self.unnormalize(g)

        if visualize:
            # reorganize our frames
            # right now, we have a list where each element correponds to a frame. and each element is a list of position of all ligands at that frame.
            # what we want is a list where each element corresponds to a single ligand. and that element will be a list of ligand positions at every frame
            lig_pos_frames = list(zip(*lig_pos_frames))
            lig_feat_frames = list(zip(*lig_feat_frames))

            return lig_pos_frames, lig_feat_frames
        
        # remove fake atoms if they were used
        if self.use_fake_atoms:
            g = self.remove_fake_atoms(g, lig_batch_idx)

        lig_pos = []
        lig_feat = []
        g = g.to('cpu')
        for g_i in dgl.unbatch(g):
            lig_pos.append(g_i.nodes['lig'].data['x_0'])
            lig_feat.append(g_i.nodes['lig'].data['h_0'])

        return lig_pos, lig_feat


    @torch.no_grad()
    def sample_given_pocket(self, rec_graph: dgl.DGLGraph, n_lig_atoms: torch.Tensor, rec_enc_batch_size: int = 32, diff_batch_size: int = 32, visualize=False):
        """Sample multiple ligands for a single binding pocket.

        Args:
            rec_graph (dgl.DGLGraph): KNN graph of just the binding pocket atoms for 1 binding pocket. Note that this is not a batched graph containing multiple receptors.
            n_lig_atoms (torch.Tensor): A 1-dimensional tensor of integers specifying how many ligand atoms there should be in each generated ligand. If the tensor is [10,12,12], then this method call will generate a ligand with 10 atoms, 2 ligands with 12 atoms.  

        Returns:
            _type_: _description_
        """        
        samples = self._sample([rec_graph], n_lig_atoms=[n_lig_atoms], rec_enc_batch_size=rec_enc_batch_size, diff_batch_size=diff_batch_size, visualize=visualize)
        lig_pos = samples[0]['positions']
        lig_feat = samples[0]['features'] 

        return lig_pos, lig_feat
        

    @torch.no_grad()
    def sample_random_sizes(self, ref_graphs: List[dgl.DGLHeteroGraph], n_replicates: int = 10, rec_enc_batch_size: int = 32, diff_batch_size: int = 32):
        
        n_receptors = len(ref_graphs)
        n_lig_atoms = self.lig_size_dist.sample((n_receptors, n_replicates))
        samples = self._sample(ref_graphs=ref_graphs, n_lig_atoms=n_lig_atoms, rec_enc_batch_size=rec_enc_batch_size, diff_batch_size=diff_batch_size)
        return samples

    def sample_p_zs_given_zt(self, s: torch.Tensor, t: torch.Tensor, g: dgl.heterograph, lig_batch_idx: torch.Tensor, kp_batch_idx: torch.Tensor):

        n_samples = g.batch_size
        device = g.device

        # compute the alpha and sigma terms that define p(z_s | z_t)
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
        sigma_s = self.sigma(gamma_s)
        sigma_t = self.sigma(gamma_t)

        # predict the noise that we should remove from this example, epsilon
        # they will each be lists containing the epsilon tensors for each ligand
        eps_h, eps_x = self.dynamics(g, t, lig_batch_idx, kp_batch_idx)

        var_terms = sigma2_t_given_s / alpha_t_given_s / sigma_t

        # expand distribution parameters by batch assignment for every ligand atom
        alpha_t_given_s = alpha_t_given_s[lig_batch_idx].view(-1, 1)
        var_terms = var_terms[lig_batch_idx].view(-1, 1)

        # compute the mean (mu) for positions/features of the distribution p(z_s | z_t)
        # this is essentially our approximation of the completely denoised ligand i think?
        # i think that's not quite correct but i think we COULD formulate sampling this way
        # -- this is how sampling is conventionally formulated for diffusion models IIRC
        # not sure why the authors settled on the alternative formulation
        mu_pos = g.nodes['lig'].data['x_0']/alpha_t_given_s - var_terms*eps_x
        mu_feat = g.nodes['lig'].data['h_0']/alpha_t_given_s - var_terms*eps_h
        
        # Compute sigma for p(zs | zt)
        sigma = sigma_t_given_s * sigma_s / sigma_t
        sigma = sigma[lig_batch_idx].view(-1, 1)

        # sample zs given the mu and sigma we just computed
        pos_noise = torch.randn(g.nodes['lig'].data['x_0'].shape, device=device)
        feat_noise = torch.randn(g.nodes['lig'].data['h_0'].shape, device=device)
        g.nodes['lig'].data['x_0'] = mu_pos + sigma*pos_noise
        g.nodes['lig'].data['h_0'] = mu_feat + sigma*feat_noise

        # remove ligand COM from system
        g = self.remove_com(g, lig_batch_idx, kp_batch_idx, com='ligand')

        return g

    def remove_fake_atoms(self, g: dgl.DGLHeteroGraph, lig_batch_idx):

        batch_size = g.batch_size
        lig_feat = g.nodes['lig'].data['h_0']
        element_idxs = torch.argmax(lig_feat, dim=1)
        fake_atom_mask = element_idxs == lig_feat.shape[1] - 1
        nodes_to_remove = torch.where(fake_atom_mask)[0]

        # check if there are no fake atoms
        if nodes_to_remove.shape[0] == 0:
            return g

        # find number of nodes that will be removed per batch
        nodes_removed_per_batch = get_nodes_per_batch(nodes_to_remove, batch_size, lig_batch_idx)

        # get batch idx of every node to be removed, convert to index pointer for segment_csr
        rm_node_batches = lig_batch_idx[nodes_to_remove]
        _, batch_segs = torch.unique_consecutive(rm_node_batches, return_counts=True)
        batch_segs = torch.concatenate([torch.zeros(1, device=g.device),  batch_segs ], dim=0)
        batch_segs = torch.cumsum(batch_segs)

        # find number of edges that will be removed per batch
        edges_removed_per_batch = {}
        for etype in ['kl', 'lk', 'll']:
            # for each edge type, count the number of those edges that each of the nodes to be remoevd is participating in
            # this is the number of edges that will be removed for each of the nodes
            # if we get the batch of each of the nodes, and then scatter_sum edges_removed_pernode by the batch assignment of every node, we get edges removed per batch
            # the issue is that scatter is not deterministic, but segment_csr is!

            edges_removed_per_node = torch.zeros_like(nodes_to_remove)
            # if ligand is dst, count in-degree for this edge type
            if etype[1] == 'l':
                edges_removed_per_node  += g.in_degrees(nodes_to_remove, etype=etype)
            # if ligand is src, count out-degree for this edge type
            if etype[0] == 'l':
                edges_removed_per_node += g.out_degrees(nodes_to_remove, etype=etype)

            edges_removed_per_batch[etype] = segment_csr(src=edges_removed_per_node, indptr=batch_segs, reduce='sum')

        batch_num_nodes, batch_num_edges = get_batch_info(g)

        # update batch information corresponding to node removal
        batch_num_nodes['lig'] = batch_num_nodes['lig'] - nodes_removed_per_batch

        for canonical_etype in batch_num_edges:
            etype = canonical_etype[1]
            if etype in edges_removed_per_batch:
                batch_num_edges[canonical_etype] = batch_num_edges[canonical_etype] - edges_removed_per_batch[etype]
        

        # remove nodes
        g.remove_nodes(nodes_to_remove, ntype='lig')

        # add batch information back into the graph
        g.set_batch_num_nodes(batch_num_nodes)
        g.set_batch_num_edges(batch_num_edges)

        return g

# noise schedules are taken from DiffSBDD: https://github.com/arneschneuing/DiffSBDD
def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


# taken from DiffSBDD: https://github.com/arneschneuing/DiffSBDD
class PredefinedNoiseSchedule(nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined
    (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]