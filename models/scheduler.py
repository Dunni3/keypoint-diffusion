from torch.optim import Optimizer
import numpy as np

class Scheduler:

    def __init__(self, 
                 optimizer: Optimizer, 
                 base_lr: float, 
                 warmup_length: float, 
                 rec_enc_loss_weight: float = 0.1,
                 rec_enc_weight_decay_midpoint: float = 0,
                 rec_enc_weight_decay_scale: float = 1,
                 restart_interval: float = 0, 
                 restart_type: str = 'linear'):
        
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.restart_interval = restart_interval
        self.restart_type = restart_type
        self.warmup_length = warmup_length

        self.rec_enc_loss_weight = rec_enc_loss_weight
        self.rec_enc_weight_decay_midpoint = rec_enc_weight_decay_midpoint
        self.rec_enc_weight_decay_scale = rec_enc_weight_decay_scale

        self.restart_marker = 0

    def step_lr(self, epoch_exact):

        if epoch_exact <= self.warmup_length:
            self.optimizer.param_groups[0]['lr'] = self.base_lr*epoch_exact/self.warmup_length

        
    def get_rec_enc_weight(self, epoch_exact):

        if self.rec_enc_weight_decay_midpoint == 0:
            return self.ref_enc_loss_weight
        
        midpoint = self.rec_enc_weight_decay_midpoint 
        scale = self.rec_enc_weight_decay_scale
        coeff = 1 - 1 / (1 + np.exp(-(epoch_exact - midpoint)*scale))

        return coeff*self.rec_enc_loss_weight
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
