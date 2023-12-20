import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass

@dataclass 
class Adam:
    lr: float = 1.0
    weight_decay: float = 0.0

@dataclass
class LBFGS:
    lr: float = 1.0

@dataclass
class Optimizer:
    type: str = 'adam'
    adam: Adam = Adam()
    lbfgs: LBFGS = LBFGS()