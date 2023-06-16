import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass
class Pytorch3d:
    blur_radius: float = 0.0
    faces_per_pixel: int = 5
    light_location: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0, -0.5]])

@dataclass
class Renderer:
    type: str = 'pytorch3d'
    image_height: int = 224
    image_width: int = 224
    mesh_color: str = 'light_blue'

    # list renderers here
    pytorch3d: Pytorch3d = Pytorch3d()