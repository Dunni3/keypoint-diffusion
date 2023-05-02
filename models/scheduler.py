from torch.optim import Optimizer
import numpy as np
from pathlib import Path
from utils import save_model
from models.ligand_diffuser import LigandDiffuser

class Scheduler:

    def __init__(self,
                 model: LigandDiffuser,
                 optimizer: Optimizer, 
                 base_lr: float,
                 output_dir: Path, 
                 warmup_length: float = 0, 
                 rec_enc_loss_weight: float = 0.1,
                 rec_enc_weight_decay_midpoint: float = 0,
                 rec_enc_weight_decay_scale: float = 1,
                 restart_interval: float = 0, 
                 restart_type: str = 'linear'):
        
        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.restart_interval = restart_interval
        self.restart_type = restart_type
        self.warmup_length = warmup_length
        self.output_dir = output_dir

        self.rec_enc_loss_weight = rec_enc_loss_weight
        self.rec_enc_weight_decay_midpoint = rec_enc_weight_decay_midpoint
        self.rec_enc_weight_decay_scale = rec_enc_weight_decay_scale

        self.restart_marker = self.warmup_length

        if self.restart_type == 'linear':
            self.restart_fn = self.linear_restart
        elif self.restart_type == 'cosine':
            self.restart_fn = self.cosine_restart
        else:
            raise NotImplementedError

    def step_lr(self, epoch_exact):

        if epoch_exact <= self.warmup_length and self.warmup_length != 0:
            self.optimizer.param_groups[0]['lr'] = self.base_lr*epoch_exact/self.warmup_length
            return
        
        if self.restart_interval == 0:
            return
        
        # TODO: account for the case where we are not doing restarts but we are doing something to the learning rate, such as an exponential decay

        # assuming we are out of the warmup phase and we are now doing restarts
        epochs_into_interval = epoch_exact - self.restart_marker
        if epochs_into_interval < self.restart_interval: # if we are within a restart interval
            self.optimizer.param_groups[0]['lr'] = self.restart_fn(epochs_into_interval)
        elif epochs_into_interval >= self.restart_interval:
            self.restart_marker = epoch_exact
            self.optimizer.param_groups[0]['lr'] = self.restart_fn(0)
            # save model on restart
            model_file = self.output_dir / f'model_on_restart_{epoch_exact:.0f}.pt'
            save_model(self.model, model_file)


    def get_rec_enc_weight(self, epoch_exact):

        if self.rec_enc_weight_decay_midpoint == 0:
            return self.rec_enc_loss_weight
        
        midpoint = self.rec_enc_weight_decay_midpoint 
        scale = self.rec_enc_weight_decay_scale
        coeff = 1 - 1 / (1 + np.exp(-(epoch_exact - midpoint)*scale))

        return coeff*self.rec_enc_loss_weight
    
    def linear_restart(self, epochs_into_interval):
        new_lr = -1.0*self.base_lr*epochs_into_interval/self.restart_interval + self.base_lr
        return new_lr

    def cosine_restart(self, epochs_into_interval):
        new_lr = 0.5*self.base_lr*(1+np.cos(epochs_into_interval*np.pi/self.restart_interval))
        return new_lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

        
