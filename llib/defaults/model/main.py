import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from .regressors import Regressor
from .diffusion import Diffusion
from .optimizations import Optimization


@dataclass
class Model():

    regressor: Regressor = Regressor()
    optimization: Optimization = Optimization()
    diffusion: Diffusion = Diffusion()

conf = OmegaConf.structured(Model)