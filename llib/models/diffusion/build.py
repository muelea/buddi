from .respace import SpacedDiffusion, space_timesteps
import numpy as np 
import math 
from loguru import logger as guru

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def build_diffusion(
    steps=1000,
    noise_schedule="linear",
    rescale_timesteps=False,
    timestep_respacing="",
    loss_type="custom",
    model_mean_type='x_start',
    model_var_type='fixed_large',
):  
    # some logging
    guru.info(f"Building diffusion model with: \
               {steps} steps \
               noise schedule {noise_schedule} \
               rescale timesteps {rescale_timesteps} \
               timestep respacing {timestep_respacing} \
               loss type {loss_type} \
               model mean type {model_mean_type} \
               model var type {model_var_type}"
    )

    # respace time steps
    if not timestep_respacing:
        timestep_respacing = [steps]
    spaced_timesteps = space_timesteps(steps, timestep_respacing)

    # betas schedule
    betas = get_named_beta_schedule(noise_schedule, steps)

    diffusion = SpacedDiffusion(
        use_timesteps=spaced_timesteps,
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

    return diffusion