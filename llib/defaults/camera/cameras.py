import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass

@dataclass
class Perspective:
    # differentiable perspective camera
    afov_horizontal: float = 60
    pitch: float = 0
    yaw: float = 0
    roll: float = 0
    tx: float = 0
    ty: float = 0
    tz: float = 0
    iw: int = 224
    ih: int = 224