import torch
import numpy as np
from tqdm import tqdm
from DDPM import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt,
    uncond_prompt = None,
    input_image = None,
    strength = 0.8,
    do_cfg = True,
    cfg_scale = 7.5,
    sampler_name = 'ddpm',
    n_interference_steps = 50,
    models = {},
    seed = None,
    device = None,
    idle_device = None,
    tokenizer = None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError('strength must be between 0 and 1')
        
        if idle_device:
            to_idle = lambda x : x.to(idle_device)
        else:
            to_idle = lambda x: x
            
        #Initialize random number generator according the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        clip   = models['clip']
        clip.tp(device)
        
        if do_cfg:
            
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding = 'max_length', max_length = 77
            ).input_ids
            
            cond_tokens = torch.tensor(cond_tokens, dtype= torch.long, device = device)
            
            cond_context = clip(cond_tokens)
            
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding = 'max_length', max_length = 77
            ).input_ids
            
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            
            uncond_context = clip(uncond_tokens)
            
            context = torch.cat([cond_context, uncond_context])
            
        else:
            
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding = 'max_length', max_length = 77
            ).input_ids
            tokens = torch.tensor(tokens, dtype = torch.long, device = device)
            context = clip(tokens)
        to_idle(clip)
        
        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_interference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")
        
        
        latents_shape = (1,4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        if input_image:
            encoder = models['encoder']
            encoder.to(device)
            
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            
            input_image_tensor = np.array(input_image_tensor)
            