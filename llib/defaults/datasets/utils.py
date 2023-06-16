import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass 
class Augmentation:
    use: bool = True # if True use augementation during training
    mirror: float = 0.5 # mirror image with prob 0.5
    noise: float = 0.4 # random image noise in [1-noise, 1+noise]
    rotation: float = 30 # random image rotation in [-rotation, +rotation]
    scale: float = 0.25 # random image scaling in [1-scale, 1+scale]
    swap: float = 0.5 # swap two people order with prob 0.5


@dataclass 
class Processing:
    normalization_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalization_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    resolution: int = 224
    use: bool = True
    load_image: bool = False