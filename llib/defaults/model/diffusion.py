from dataclasses import dataclass
from .losses import Losses 
from .optimizer import Optimizer
from typing import List
from dataclasses import field


@dataclass
class Gaussian:   
    steps: int = 1000
    noise_schedule: str = 'linear' # chose from linear, cosine
    rescale_timesteps: bool = False
    timestep_respacing: str = ""
    loss_type: str = 'custom' # chose from mse, rescaled_mse, kl, rescaled_kl, custom
    model_mean_type: str = 'start_x' # chose from start_x, epislon, previous_x
    model_var_type: str = 'fixed_large' # chose from fixed_large, fixed_small, learned_range
    

@dataclass 
class Diffusion():
    steps: int = 1000
    noise_schedule: str = 'linear' # chose from linear, cosine
    rescale_timesteps: bool = False
    timestep_respacing: str = ""
    loss_type: str = 'custom' # chose from mse, rescaled_mse, kl, rescaled_kl, custom
    model_mean_type: str = 'start_x' # chose from start_x, epislon, previous_x
    model_var_type: str = 'fixed_large' # chose from fixed_large, fixed_small, learned_range