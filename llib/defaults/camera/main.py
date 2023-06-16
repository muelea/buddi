import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from .cameras import Perspective

@dataclass 
class Camera:
    type: str = 'perspective'

    # list all camera types here
    perspective: Perspective = Perspective()

conf = OmegaConf.structured(Camera)