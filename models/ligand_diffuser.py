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
from utils import concat_graph_data

class LigandDiffuser(nn.Module):

    def __init__(self, atom_nf, rec_nf, n_keypoints: int, processed_dataset_dir: Path, n_timesteps: int = 1000, keypoint_centered=False, graph_config={},
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
            self.dynamics = LigRecDynamics(atom_nf, rec_nf, no_cg=rec_encoder_config['no_cg'], **dynamics_config)
        else:
            self.dynamics = LigRecDynamics(atom_nf, rec_nf, **dynamics_config)

        # create receptor encoder and its loss function
        self.rec_encoder = ReceptorEncoder(**graph_config, **rec_encoder_config)
        self.rec_encoder_loss_fn = ReceptorEncoderLoss(**rec_encoder_loss_config)

    def forward(self, complex_graphs: dgl.DGLHeteroGraph, interface_points: List[torch.Tensor]):
        """Computes loss."""
        # normalize values. specifically, atom features are normalized by a value of 4
        losses = {}

        complex_graphs = self.normalize(complex_graphs)

        batch_size = complex_graphs.batch_size
        device = complex_graphs.device
                
        # encode the receptor
        complex_graphs = self.rec_encoder(complex_graphs)

        # concatenate the receptor keypoints and get index tensors so we can do segmented operations on them
        kp_pos, kp_idx, kp_indptr = concat_graph_data(kp_pos)
        kp_feat = torch.concatenate(kp_feat, dim=0)

        # concatenate ligands and get indexes so we can do segemnted operations
        lig_atom_positions, lig_idx, lig_indptr = concat_graph_data(lig_atom_positions)
        lig_atom_features = torch.concatenate(lig_atom_features)

        # if we are applying the RL hinge loss, we will need to be able to put receptor atoms and the ligand into the same
        # referance frame. in order to do this, we need the initial COM of the keypoints
        if self.apply_rl_hinge_loss:
            init_kp_com = segment_csr(kp_pos, kp_indptr, reduce="mean")

        # compute receptor encoding loss
        losses['rec_encoder'] = self.rec_encoder_loss_fn(kp_pos, interface_points=interface_points)

        # remove ligand COM from receptor/ligand complex
        kp_pos, lig_atom_positions = self.remove_com(kp_pos, lig_atom_positions, kp_indptr, kp_idx, lig_indptr, lig_idx, com='ligand')

        # sample timepoints for each item in the batch
        t = torch.randint(0, self.n_timesteps, size=(batch_size,), device=device).float() # timesteps
        t = t / self.n_timesteps

        # sample epsilon for each ligand
        eps = {
            'h':torch.randn(lig_atom_features.shape, device=device),
            'x':torch.randn(lig_atom_positions.shape, device=device)
        }
        
        # construct noisy versions of the ligand
        gamma_t = self.gamma(t).to(device=device)
        zt_pos, zt_feat, kp_pos = self.noised_representation(lig_atom_positions, lig_atom_features, kp_pos, lig_idx, lig_indptr, kp_idx, kp_indptr, gamma_t)

        # predict the noise that was added
        if self.apply_rl_hinge_loss:
            unbatch_eps = True
        else:
            unbatch_eps = False

        eps_h_pred, eps_x_pred = self.dynamics(zt_pos, zt_feat, kp_pos, kp_feat, t, 
                                               unbatch_eps=unbatch_eps)

        # compute hinge loss if necessary
        if self.apply_rl_hinge_loss:
            # predict denoised ligand
            x0_pos_pred, _ = self.denoised_representation(zt_pos, zt_feat, eps_x_pred, eps_h_pred, gamma_t)

            # translate ligand back to intitial frame of reference
            _, x0_pos_pred = self.remove_com(kp_pos, x0_pos_pred, com='receptor')
            x0_pos_pred = [ x+init_kp_com[i] for i,x in enumerate(x0_pos_pred) ]

            # get atom positions for all receptors
            rec_atom_positions = [g.ndata["x_0"] for g in dgl.unbatch(rec_graphs)]

            # compute hinge loss between ligand atom position and receptor atom positions
            rl_hinge_loss = 0
            for denoised_lig_pos, rec_atom_pos in zip(x0_pos_pred, rec_atom_positions):
                rl_hinge_loss += self.rl_hinge_loss_fn(denoised_lig_pos, rec_atom_pos)

            losses['rl_hinge'] = rl_hinge_loss

            # concat eps_h_pred and eps_x_pred along batch dim 
            # this is so that the computaton of l2 loss is unaffected
            eps_x_pred = torch.concat(eps_x_pred, dim=0)
            eps_h_pred = torch.concat(eps_h_pred, dim=0)

        # concatenate the added the noises together
        eps_x = torch.concat([ eps_dict['x'] for eps_dict in eps_batch ], dim=0)
        eps_h = torch.concat([ eps_dict['h'] for eps_dict in eps_batch ], dim=0)

        # compute l2 loss on noise
        if self.use_fake_atoms:
            real_atom_mask = torch.concat([ ~(lig_feat[:, -1].bool()) for lig_feat in lig_atom_features ])[:, None]
            n_real_atoms = real_atom_mask.sum()
            n_x_loss_terms = n_real_atoms*3
            x_loss = ((eps_x - eps_x_pred)*real_atom_mask).square().sum() # mask out loss on predicted position of fake atoms
        else:
            x_loss = ((eps_x - eps_x_pred)).square().sum()
            n_x_loss_terms = eps_x.numel()

        h_loss = (eps_h - eps_h_pred).square().sum()
        losses['l2'] = (x_loss + h_loss) / (n_x_loss_terms + eps_h.numel())

        losses['pos'] = x_loss / n_x_loss_terms
        losses['feat'] = h_loss / eps_h.numel()

        return losses
    
    def normalize(self, complex_graphs: dgl.DGLHeteroGraph):
        complex_graphs.nodes['lig'].data['h_0'] = complex_graphs.nodes['lig'].data['h_0'] / self.lig_feat_norm_constant
        return complex_graphs

    def unnormalize(self, complex_graphs: dgl.DGLHeteroGraph):
        complex_graphs.nodes['lig'].data['h_0'] = complex_graphs.nodes['lig'].data['h_0'] * self.lig_feat_norm_constant
        return complex_graphs

    def remove_com(self, kp_pos: torch.Tensor, lig_pos: torch.Tensor, kp_indptr: torch.Tensor, kp_idx: torch.Tensor, lig_indptr: torch.Tensor, lig_idx: torch.Tensor, com: str = None):
        """Remove center of mass from ligand atom positions and receptor keypoint positions.

        This method can remove either the ligand COM, protein COM or the complex COM.

        Args:
            kp_pos (torch.Tensor): A tensor of shape (n_kp_nodes, 3) containing keypoint or receptor atom positions for a batch concatenated alone the n_kp_nodes dimension.
            lig_pos (torch.Tensor): A tensor of shape (n_lig_nodes, 3) containing ligand positions for a batch concatenated along the n_lig_nodes dimension
            kp_indptr (torch.Tensor): Index pointer array for keypoints required for segmented ops.
            kp_idx (torch.Tensor): Index array specifying which graph each keypoint belongs to, requried for segmented ops.
            lig_indptr (torch.Tensor): Same as kp_indptr but for ligand atoms.
            lig_idx (torch.Tensor): Same as kp_idx but for ligand atoms.
            com (str, optional): Specifies which center of mass to remove from the system. Options are 'ligand', 'receptor', or None. If None, the COM of the ligand/receptor complex will be removed. Defaults to None.

        Raises:
            NotImplementedError: if com=None because I never implemented this feature and haven't needed it
            ValueError: If an unsupported values is passed for comargument.

        Returns:
            torch.Tensor: Receptor keypoints with COM removed.
            torch.Tensor: Ligand atom positions with COM removed.
        """               
        if com is None:
            raise NotImplementedError('removing COM of receptor/ligand complex not implemented')
        elif com == 'ligand':
            com = segment_csr(lig_pos, lig_indptr, reduce='mean')
        elif com == 'receptor':
            com = segment_csr(kp_pos, kp_indptr, reduce='mean')
        else:
            raise ValueError(f'invalid value for com: {com=}')

        com_free_lig = lig_pos - com[lig_idx]
        com_free_kp = kp_pos - com[kp_idx]
        return com_free_kp, com_free_lig

    def noised_representation(self, lig_pos: torch.Tensor, lig_feat: torch.Tensor, rec_pos: torch.Tensor, lig_idx, lig_indptr, rec_idx, rec_indptr,
                              eps: Dict[str, torch.Tensor], gamma_t: torch.Tensor):
        alpha_t = self.alpha(gamma_t)[lig_idx]
        sigma_t = self.sigma(gamma_t)[lig_idx]

        zt_feat = alpha_t*lig_feat + sigma_t*eps['h']

        # remove ligand COM from the system
        rec_pos, zt_pos = self.remove_com(rec_pos, zt_pos, rec_indptr, rec_idx, lig_indptr, lig_idx, com='ligand')
        
        return zt_pos, zt_feat, rec_pos
    
    def denoised_representation(self, zt_pos, zt_feat, eps_x_pred, eps_h_pred, gamma_t):
        # assuming the input ligand COM is zero, we compute the denoised verison of the ligand
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma(gamma_t)

        x0_pos, x0_feat = [], []
        for i in range(len(gamma_t)):
            x0_pos.append( (zt_pos[i] - sigma_t[i]*eps_x_pred[i])/alpha_t[i] )
            x0_feat.append( (zt_feat[i] - sigma_t[i]*eps_h_pred[i])/alpha_t[i] )

        return x0_pos, x0_feat

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

    def encode_receptors(self, receptors: List[dgl.DGLGraph]):

        device = receptors[0].device
        n_receptors = len(receptors)

        # compute initial receptor atom COM
        init_rec_atom_com = [ g.ndata["x_0"].mean(dim=0, keepdim=True) for g in receptors ]

        # get keypoints positions/features
        kp_pos, kp_feat = self.rec_encoder(dgl.batch(receptors))

        # get initial keypoint center of mass
        init_kp_com = [ x.mean(dim=0, keepdim=True) for x in kp_pos ]

        # remove (receptor atom COM, or keypoint COM) from receptor keypoints
        sampling_com = init_rec_atom_com
        kp_pos = [  kp_pos[i] - sampling_com[i] for i in range(len(kp_pos)) ]

        return kp_pos, kp_feat, init_rec_atom_com, init_kp_com

    
    @torch.no_grad()
    def _sample(self, receptors: List[dgl.DGLGraph], n_lig_atoms: List[List[int]], rec_enc_batch_size: int = 32, diff_batch_size: int = 32, visualize=False) -> List[List[Dict[str, torch.Tensor]]]:
        """Sample multiple receptors with multiple ligands per receptor.

        Args:
            receptors (List[dgl.DGLGraph]): A list containing a DGL graph of each receptor that is to be sampled.
            n_lig_atoms (List[List[int]]): A list that contains a list for each receptor. Each nested list contains integers that each specify the number of atoms in a ligand.
            rec_enc_batch_size (int, optional): Batch size for forward passes through receptor encoder. Defaults to 32.
            diff_batch_size (int, optional): Batch size for forward passes through denoising model. Defaults to 32.

        Returns:
            List[Dict[str, torch.Tensor]]: A list of length len(receptors). Each element of this list is a dictionary with keys "positions" and "features". The values are lists of tensors, one tensor per ligand. 
        """        

        device = receptors[0].device
        n_receptors = len(receptors)
        
        # encode the receptors
        kp_pos_src = []
        kp_feat_src = []
        init_rec_atom_com_src = []
        init_kp_com_src = []
        for batch_idx in range(ceil(n_receptors / rec_enc_batch_size)):

            # determine number of receptors that will be in this batch
            n_samples_batch = min(rec_enc_batch_size, n_receptors - len(kp_pos_src))

            # select receptors for this batch
            batch_idx_start = batch_idx*rec_enc_batch_size
            batch_idx_end = batch_idx_start + n_samples_batch
            batch_receptors = receptors[batch_idx_start:batch_idx_end]

            # encode receptors and get COM of receptor atoms and keypoint positions
            batch_kp_pos, batch_kp_feat, batch_init_rec_atom_com, batch_init_kp_com = self.encode_receptors(batch_receptors)

            # concat the encoded receptor information from this batch with those from previous batches
            init_rec_atom_com_src.extend(batch_init_rec_atom_com)
            init_kp_com_src.extend(batch_init_kp_com)
            kp_pos_src.extend(batch_kp_pos)
            kp_feat_src.extend(batch_kp_feat)

        # generate list of receptor/ligand pairs
        kp_pos, kp_feat = [], []
        init_rec_atom_com = []
        init_kp_com = []
        n_lig_atoms_flattened = [] # this will be a list of integers, each integer is the number of ligand atoms for a complex
        for rec_idx in range(n_receptors):
            
            n_ligands = len(n_lig_atoms[rec_idx]) # number of ligands to be sampled for this receptor

            n_lig_atoms_flattened.extend(n_lig_atoms[rec_idx]) # build n_lig_atoms_flattened

            # extend kp_pos, kp_feat, and COM lists by the values for this receptor copied n_ligand times
            kp_pos.extend([ kp_pos_src[rec_idx].detach().clone() for _ in range(n_ligands) ])
            kp_feat.extend([ kp_feat_src[rec_idx].detach().clone() for _ in range(n_ligands) ])
            init_rec_atom_com.extend([ init_rec_atom_com_src[rec_idx].detach().clone() for _ in range(n_ligands) ])
            init_kp_com.extend([ init_kp_com_src[rec_idx].detach().clone() for _ in range(n_ligands) ])


        # proceed to batched sampling
        n_complexes = len(kp_pos)
        n_complexes_sampled = 0
        lig_pos, lig_feat = [], []
        for batch_idx in range(ceil(n_complexes / diff_batch_size)):

            # determine number of complexes that will be in this batch
            n_samples_batch = min(diff_batch_size, n_complexes - n_complexes_sampled)

            start_idx = batch_idx*diff_batch_size
            end_idx = start_idx + n_samples_batch

            batch_kp_pos = kp_pos[start_idx:end_idx]
            batch_kp_feat = kp_feat[start_idx:end_idx]
            batch_n_atoms = n_lig_atoms_flattened[start_idx:end_idx]
            batch_init_kp_com = init_kp_com[start_idx:end_idx]
            batch_init_rec_atom_com = init_rec_atom_com[start_idx:end_idx]

            batch_lig_pos, batch_lig_feat = self.sample_from_encoded_receptors(batch_kp_pos, batch_kp_feat, batch_init_rec_atom_com, batch_init_kp_com, batch_n_atoms, visualize=visualize)
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

    @torch.no_grad()
    def sample_from_encoded_receptors(self, kp_pos: List[torch.Tensor], kp_feat: List[torch.Tensor], init_atom_com: List[torch.Tensor], init_kp_com: List[torch.Tensor], n_lig_atoms: List[int], visualize=False):

        device = kp_pos[0].device
        n_complexes = len(kp_pos)

        # sample initial positions/features of ligands
        lig_pos, lig_feat = [], []
        for complex_idx in range(n_complexes):
            lig_pos.append(torch.randn((n_lig_atoms[complex_idx], 3), device=device)) 
            lig_feat.append(torch.randn((n_lig_atoms[complex_idx], self.n_lig_features), device=device))

        if visualize:
            init_kp_com_cpu = [ x.detach().cpu() for x in init_kp_com ]
            # convert positions and features to cpu
            # convert positions to input frame of reference: remove current kp com and add original init kp com
            # note that this function assumes that the keypoints passed as arguments have the keypoint COM removed from them already, so all we need to do is add back in the initial keypoint COM
            lig_pos_frames = [ [ x.detach().cpu() + init_kp_com_cpu[i] for i, x in enumerate(lig_pos) ] ]
            lig_feat_frames = [ [ x.detach().cpu() for x in lig_feat ] ]

        # remove ligand com from every receptor/ligand complex
        kp_pos, lig_pos = self.remove_com(kp_pos, lig_pos, com='ligand')

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.n_timesteps)):
            s_arr = torch.full(size=(n_complexes,), fill_value=s, device=device)
            t_arr = s_arr + 1
            s_arr = s_arr / self.n_timesteps
            t_arr = t_arr / self.n_timesteps

            lig_feat, lig_pos, kp_pos = self.sample_p_zs_given_zt(s_arr, t_arr, kp_pos, kp_feat, lig_pos, lig_feat)

            if visualize:

                # convert keypoints positions, ligand atom positions, and ligand features to the cpu
                kp_pos_cpu = [ x.detach().cpu() for x in kp_pos ]
                frame_pos = [ x.detach().cpu() for x in lig_pos ]
                frame_feat = [ x.detach().cpu() for x in lig_feat ]

                # move ligand atoms back to initial frame of reference
                frame_pos = [ x - kp_pos_cpu[i].mean(dim=0, keepdim=True) + init_kp_com_cpu[i]  for i,x in enumerate(frame_pos) ]
                lig_pos_frames.append(frame_pos)
                lig_feat_frames.append(frame_feat)


        # remove keypoint COM from system after generation
        kp_pos, lig_pos = self.remove_com(kp_pos, lig_pos, com='receptor')

        # TODO: model P(x0 | x1)?

        # add initial keypoint COM to system, bringing us back into the input frame of reference
        for i in range(n_complexes):
            lig_pos[i] += init_kp_com[i]
            # kp_pos[i] += init_kp_com[i]

        # unnormalize features
        lig_pos, lig_feat = self.unnormalize(lig_pos, lig_feat)

        if visualize:

            # remove fake atoms from all frames if they're being used
            if self.use_fake_atoms:
                new_pos_and_feat = [ self.remove_fake_atoms(lig_pos_frames[frame_idx], lig_feat_frames[frame_idx]) for frame_idx in range(len(lig_pos_frames)) ]
                lig_pos_frames, lig_feat_frames = list(map(list, zip(*new_pos_and_feat)))

            # reorganize our frames
            # right now, we have a list where each element correponds to a frame. and each element is a list of position of all ligands at that frame.
            # what we want is a list where each element corresponds to a single ligand. and that element will be a list of ligand positions at every frame
            lig_pos_frames = list(zip(*lig_pos_frames))
            lig_feat_frames = list(zip(*lig_feat_frames))

            return lig_pos_frames, lig_feat_frames
        
        # remove fake atoms if they were used
        if self.use_fake_atoms:
            lig_pos, lig_feat = self.remove_fake_atoms(lig_pos, lig_feat)

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
    def sample_random_sizes(self, receptors: List[dgl.DGLGraph], n_replicates: int = 10, rec_enc_batch_size: int = 32, diff_batch_size: int = 32):
        
        n_receptors = len(receptors)
        n_lig_atoms = self.lig_size_dist.sample((n_receptors, n_replicates))
        samples = self._sample(receptors=receptors, n_lig_atoms=n_lig_atoms, rec_enc_batch_size=rec_enc_batch_size, diff_batch_size=diff_batch_size)
        return samples

    def sample_p_zs_given_zt(self, s, t, rec_pos: List[torch.Tensor], rec_feat: List[torch.Tensor], zt_pos: List[torch.Tensor], zt_feat: List[torch.Tensor]):

        n_samples = len(s)
        device = rec_pos[0].device

        # compute the alpha and sigma terms that define p(z_s | z_t)
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
        sigma_s = self.sigma(gamma_s)
        sigma_t = self.sigma(gamma_t)

        # predict the noise that we should remove from this example, epsilon
        # they will each be lists containing the epsilon tensors for each ligand
        eps_h, eps_x = self.dynamics(zt_pos, zt_feat, rec_pos, rec_feat, t, unbatch_eps=True)

        var_terms = sigma2_t_given_s / alpha_t_given_s / sigma_t

        # compute the mean (mu) for positions/features of the distribution p(z_s | z_t)
        # this is essentially our approximation of the completely denoised ligand i think?
        # i think that's not quite correct but i think we COULD formulate sampling this way
        # -- this is how sampling is conventionally formulated for diffusion models IIRC
        # not sure why the authors settled on the alternative formulation
        mu_pos = []
        mu_feat = []
        for i in range(n_samples):
            mu_pos_i = zt_pos[i]/alpha_t_given_s[i] - var_terms[i]*eps_x[i]
            mu_feat_i = zt_feat[i]/alpha_t_given_s[i] - var_terms[i]*eps_h[i]

            mu_pos.append(mu_pos_i)
            mu_feat.append(mu_feat_i)
        
        # Compute sigma for p(zs | zt)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # sample zs given the mu and sigma we just computed
        zs_pos = []
        zs_feat = []
        for i in range(n_samples):
            pos_noise = torch.randn(zt_pos[i].shape, device=device)
            feat_noise = torch.randn(zt_feat[i].shape, device=device)
            zs_pos.append( mu_pos[i] + sigma[i]*pos_noise )
            zs_feat.append(mu_feat[i] + sigma[i]*feat_noise)

        # remove ligand COM from system
        rec_pos, zs_pos = self.remove_com(rec_pos, zs_pos, com='ligand')

        return zs_feat, zs_pos, rec_pos

    def remove_fake_atoms(self, lig_pos: List[torch.Tensor], lig_feat: List[torch.Tensor]):
        # remove atoms marked as the "not atom" type
        for idx, (lig_pos_i, lig_feat_i) in enumerate(zip(lig_pos, lig_feat)):
            element_idxs = torch.argmax(lig_feat_i, dim=1)

            # remove atoms marked as the "not atom" type
            real_atom_mask = element_idxs != lig_feat_i.shape[1] - 1
            lig_pos_i = lig_pos_i[real_atom_mask] # remove fake atoms from positions
            lig_feat_i = lig_feat_i[real_atom_mask][:, :-1] # remove fake from features and slice off the "no atom" type
            lig_pos[idx] = lig_pos_i
            lig_feat[idx] = lig_feat_i
        return lig_pos, lig_feat

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