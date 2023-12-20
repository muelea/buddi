import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from .losses import Losses
from .optimizer import Optimizer
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class SmplifyDc:
    use_contact: bool = "True"


@dataclass
class HHCS:
    use_contact: bool = "True"
    max_iters: List[int] = field(default_factory=lambda: [100, 100])
    num_prev_steps: int = 100
    slope_tol: float = -0.00001


@dataclass
class Optimization:
    type: str = "smplifydc"
    print_loss: bool = True
    render_iters: bool = False
    use_gt_contact_map: bool = True
    use_vae: bool = False
    pretrained_vae_cfg: str = ""
    pretrained_vae_ckpt: str = ""
    use_diffusion: bool = False
    pretrained_diffusion_model_cfg: str = ""
    pretrained_diffusion_model_ckpt: str = ""
    sds_type: str = "fixed"  # can be range, fixed, or adaptive
    # sds t for selected sds type
    sds_t_fixed: int = 20  # the t in the diffusion step
    sds_t_range: List[int] = field(
        default_factory=lambda: [25, 75]
    )  # sample from t in the diffusion step
    sds_t_adaptive_i: List[float] = field(
        default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2]
    )  # the t in the diffusion step
    sds_t_adaptive_t: List[int] = field(
        default_factory=lambda: [100, 80, 60, 40, 20]
    )  # the t in the diffusion step

    optimizer: Optimizer = Optimizer()
    losses: Losses = Losses()

    # list all optimizations here
    smplifydc: SmplifyDc = SmplifyDc()
    hhcs: HHCS = HHCS()
