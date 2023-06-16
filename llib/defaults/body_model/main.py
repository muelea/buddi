import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from .guhm import GUHM 
from .smpl import SMPL 
from .smplh import SMPLH 
from .smplx import SMPLX 

@dataclass
class BodyModel:
    type: str = 'smplx'
    smpl_family_folder: str = 'essentials/body_models'
    smpl: SMPL = SMPL()
    smplh: SMPLH = SMPLH()
    smplx: SMPLX = SMPLX()

conf = OmegaConf.structured(BodyModel)