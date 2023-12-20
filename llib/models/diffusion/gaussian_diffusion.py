"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
"""

import numpy as np
import torch

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.one = torch.tensor(1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def p_sample_ddim(self, x_start, t, prev_t, x_t, eta=0.0):
        """
        Denoise the data for a given number of diffusion steps.
        Sample from p(x_{t-1} | x_t, x_0).
        ------------
        x_start: model output (denoised model prediction) (x_0)
        t: the current timestep, t
        prev_t: the previous timestep, t-1
        x_t: the current noisy input (x_t)
        """
        # 1. compute alphas, betas
        alpha_prod_t = _extract_into_tensor(self.alphas_cumprod, t, x_start.shape)
        alpha_prod_t_prev = _extract_into_tensor(self.alphas_cumprod, prev_t, x_start.shape) \
            if prev_t is not None else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** (0.5)

        pred_epsilon = (x_t - alpha_prod_t ** (0.5) * x_start) / beta_prod_t ** (0.5)

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        prev_sample = alpha_prod_t_prev ** (0.5) * x_start + pred_sample_direction

        # 6. Add noise        
        if eta > 0:
            variance_noise = torch.randn_like(x_start)
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance
        
        if not eta > 0:
            variance = torch.zeros_like(x_start)

        return prev_sample, variance
    
        
    def p_sample_ddpm(self, x_start, t, prev_t, x_t):
        """
        Denoise the data for a given number of diffusion steps.
        Sample from p(x_{t-1} | x_t, x_0).
        ------------
        x_start: model output (denoised model prediction) (x_0)
        t: the current timestep, t
        prev_t: the previous timestep, t-1
        x_t: the current noisy input (x_t)
        """
        # 1. compute alphas, betas
        alpha_prod_t = _extract_into_tensor(self.alphas_cumprod, t, x_start.shape)
        alpha_prod_t_prev = _extract_into_tensor(self.alphas_cumprod, prev_t, x_start.shape) \
            if prev_t is not None else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * x_start + current_sample_coeff * x_t

        # 6. Add noise
        variance = 0
        if t[0] > 0:
            current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

            # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
            # and sample from it to get previous sample
            # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

            # we always take the log of variance, so clamp it to ensure it's not 0
            variance = torch.clamp(variance, min=1e-20)

            # hacks - were probably added for training stability
            if self.model_var_type == "fixed_small":
                variance = variance
            # for rl-diffuser https://arxiv.org/abs/2205.09991
            elif self.model_var_type == "fixed_small_log":
                variance = torch.log(variance)
                variance = torch.exp(0.5 * variance)
            elif self.model_var_type == "fixed_large":
                variance = current_beta_t
            elif self.model_var_type == "fixed_large_log":
                # Glide max_log
                variance = torch.log(current_beta_t)

            variance_noise = torch.randn_like(x_start)
            pred_prev_sample = pred_prev_sample + variance * variance_noise

            return pred_prev_sample, variance
        
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
