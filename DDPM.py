import torch

import numpy as np

class DDPMSampler:
    #DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
    
    def __init__(self, generator : torch.Generator, num_training_steps = 1000, beta_start :float = 0.00085, beta_end : float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) **2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = 0)
        self.one = torch.tensor(1.0)
        
        self.generator = generator
        
        self.num_train_timestamps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1]. copy())
        
    def set_inference_timesteps(self, num_inference_steps = 50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timestamps // self.num_inference_steps
        timesteps = (np.arange(0,num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        
    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_train_timestamps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
           
           
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        
        variance = torch.clamp(variance, min = 1e-20)
        
        return variance
    
    
    def set_strength(self, strength = 1):
        """
        
        Setting how much noise to add to the image
        Strength ~ 1 = output will be furthest away from the input image
        Strength ~ 0 = output will be the closer to the input image
    
        """
        
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
        
    def step(self, timestep: int, latents : torch.Tensor, model_output = torch.Tensor):
        t = timestep
        prev_t = self. _get_previous_timestep(t)
        
        
        #Commute all the alphas and the betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t/ alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample+ current_sample_coeff * latents
        
        
        #Add noise
        
        variance = 0
        
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape,generator= self.generator,device = device, dtype = model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise
            
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample

    def add_noise (
        self, 
        original_samples : torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        timesteps = timesteps.to(original_samples.deivce)
        
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        
        noise = torch.randn(original_samples.shape,generator=self.generator,device = original_samples.device,dtype = original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
        