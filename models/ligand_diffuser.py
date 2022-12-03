import torch.nn as nn
import torch
import numpy as np

from models.dynamics import LigRecDynamics

class LigandDiffuser(nn.Module):

    def __init__(self, atom_nf, rec_nf, n_timesteps: int = 1000):
        super().__init__()

        # TODO: add default keyword arguments from LigRecDynamics to config file
        self.n_timesteps = n_timesteps
        self.gamma = PredefinedNoiseSchedule(noise_schedule='polynomial_2', timesteps=n_timesteps, precision=1e-4)
        self.dynamics = LigRecDynamics(atom_nf, rec_nf)

    def forward(self, lig_atom_positions, lig_atom_features, rec_pos, rec_feat):
        """Computes loss."""
        # input: ligand atom positions/features + rec atom or keypoint positions/features

        batch_size = len(lig_atom_positions)

        # sample timepoints for each item in the batch
        t = torch.randint(0, self.n_timesteps, size=(len(lig_atom_positions),)).float() # timesteps
        t = t / self.n_timesteps

        # sample epsilon for each ligand
        eps_batch = []
        for i in range(batch_size):
            eps = {
                'h': torch.randn(lig_atom_features[i].shape),
                'x': self.sample_com_free(lig_atom_positions[i].shape)
            }
            eps_batch.append(eps)
        
        # construct noisy versions of the ligand
        gamma_t = self.gamma(t)
        zt_pos, zt_feat = self.noised_representation(lig_atom_positions, lig_atom_features, eps_batch, gamma_t)

        # predict the noise that was added
        eps_h_pred, eps_x_pred = self.dynamics(zt_pos, zt_feat, rec_pos, rec_feat, t)

        # concatenate the added the noises together
        eps_x = torch.concat([ eps_dict['x'] for eps_dict in eps_batch ], dim=0)
        eps_h = torch.concat([ eps_dict['h'] for eps_dict in eps_batch ], dim=0)

        # compute l2 loss on noise
        x_loss = (eps_x - eps_x_pred).square().sum()
        h_loss = (eps_h - eps_h_pred).square().sum()
        l2_loss = x_loss + h_loss

        return l2_loss

    def sample_com_free(self, shape):
        eps = torch.randn(shape)
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