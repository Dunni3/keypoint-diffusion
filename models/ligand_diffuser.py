import torch.nn as nn
import torch
import numpy as np
from typing import List
import dgl
import torch.nn.functional as fn

from models.dynamics import LigRecDynamics
from models.receptor_encoder import ReceptorEncoder
from losses.rec_encoder_loss import ReceptorEncoderLoss

class LigandDiffuser(nn.Module):

    def __init__(self, atom_nf, rec_nf, n_timesteps: int = 1000, 
    dynamics_config = {}, rec_encoder_config = {}, rec_encoder_loss_config= {}):
        super().__init__()

        self.n_lig_features = atom_nf
        self.n_kp_feat = rec_nf

        # TODO: add default keyword arguments from LigRecDynamics to config file
        self.n_timesteps = n_timesteps
        self.gamma = PredefinedNoiseSchedule(noise_schedule='polynomial_2', timesteps=n_timesteps, precision=1e-4)
        self.dynamics = LigRecDynamics(atom_nf, rec_nf, **dynamics_config)

        self.rec_encoder = ReceptorEncoder(**rec_encoder_config)
        self.rec_encoder_loss_fn = ReceptorEncoderLoss(**rec_encoder_loss_config)

    def forward(self, rec_graphs, lig_atom_positions, lig_atom_features):
        """Computes loss."""
        # TODO: normalize values. specifically, atom features are normalized by a value of 4

        batch_size = len(lig_atom_positions)
        device = lig_atom_positions[0].device

        # encode the receptor
        rec_pos, rec_feat = self.rec_encoder(rec_graphs)

        # compute receptor encoding loss
        ot_loss = self.rec_encoder_loss_fn(rec_pos, rec_graphs)

        # sample timepoints for each item in the batch
        t = torch.randint(0, self.n_timesteps, size=(len(lig_atom_positions),), device=device).float() # timesteps
        t = t / self.n_timesteps

        # sample epsilon for each ligand
        eps_batch = []
        for i in range(batch_size):
            eps = {
                'h': torch.randn(lig_atom_features[i].shape, device=device),
                'x': self.sample_com_free(lig_atom_positions[i].shape, device=device)
            }
            eps_batch.append(eps)
        
        # construct noisy versions of the ligand
        gamma_t = self.gamma(t).to(device=device)
        zt_pos, zt_feat = self.noised_representation(lig_atom_positions, lig_atom_features, eps_batch, gamma_t)

        # predict the noise that was added
        eps_h_pred, eps_x_pred = self.dynamics(zt_pos, zt_feat, rec_pos, rec_feat, t)

        # concatenate the added the noises together
        eps_x = torch.concat([ eps_dict['x'] for eps_dict in eps_batch ], dim=0)
        eps_h = torch.concat([ eps_dict['h'] for eps_dict in eps_batch ], dim=0)

        # compute l2 loss on noise
        x_loss = (eps_x - eps_x_pred).square().sum() / eps_x.numel()
        h_loss = (eps_h - eps_h_pred).square().sum() / eps_h.numel()
        l2_loss = x_loss + h_loss
        l2_loss = l2_loss

        return l2_loss, ot_loss

    def sample_com_free(self, shape, device):
        eps = torch.randn(shape, device=device)
        eps = eps - eps.mean()
        return eps

    def noised_representation(self, lig_pos, lig_feat, eps_batch, gamma_t):
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma(gamma_t)

        zt_pos, zt_feat = [], []
        for i in range(len(gamma_t)):
            zt_pos.append(alpha_t[i]*lig_pos[i] + sigma_t[i]*eps_batch[i]['x'])
            zt_feat.append(alpha_t[i]*lig_feat[i] + sigma_t[i]*eps_batch[i]['h'])
        
        return zt_pos, zt_feat

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


    @torch.no_grad()
    def sample_given_pocket(self, rec_graph: dgl.DGLGraph, n_lig_atoms: torch.Tensor):

        device = rec_graph.device
        n_samples = len(n_lig_atoms)

        # encode the receptor
        rec_pos, rec_feat = self.rec_encoder(rec_graph)

        # copy the receptor encoding so a separate graph can be made for each ligand
        rec_pos = [ rec_pos[0].detach().clone() for _ in range(n_samples) ]
        rec_feat = [ rec_feat[0].detach().clone() for _ in range(n_samples) ]

        # sample initial ligand positions/features
        lig_pos, lig_feat = [], []
        for i in range(n_samples):
            lig_pos.append(self.sample_com_free(shape=(n_lig_atoms[i], 3), device=device))
            lig_feat.append(torch.randn((n_lig_atoms[i], self.n_lig_features), device=device))

        # TODO: note that at the time of writing this, I am only sampling from receptors which have been preprocessed
        # by my process_crossdocked.py script. Specifically, this means that all receptors are positioned such that
        # the ligand COM is at 0. This is why we can sample initial ligand positions using the standard normal.
        # We will have to do something different in the future, as I think this gives it a weird bias, and to make this code
        # so that you can sample any given pocket no matter where the coordinates are
        # in other words, this is an assumption of the code in this function that i would like to not rely on in the future

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.n_timesteps)):
            s_arr = torch.full(size=(n_samples,), fill_value=s, device=device)
            t_arr = s_arr + 1
            s_arr = s_arr / self.n_timesteps
            t_arr = t_arr / self.n_timesteps

            lig_feat, lig_pos = self.sample_p_zs_given_zt(s_arr, t_arr, rec_pos, rec_feat, lig_pos, lig_feat)

        return lig_pos, lig_feat

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
            pos_noise = self.sample_com_free(shape=zt_pos[i].shape, device=device)
            feat_noise = torch.randn(zt_feat[i].shape, device=device)
            zs_pos.append( mu_pos[i] + sigma[i]*pos_noise )
            zs_feat.append(mu_feat[i] + sigma[i]*feat_noise)

        return zs_feat, zs_pos



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