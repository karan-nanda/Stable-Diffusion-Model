import torch

import numpy as np

class DDPMSampler:
    
    def __init__(self, generator : torch.Generator, num_training_steps = 1000, beta_start :float = 0.00085, beta_end : float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) **2
        