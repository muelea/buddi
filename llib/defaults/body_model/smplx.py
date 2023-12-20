import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Optional, Dict, Union
from .smpl_family_params import Expression, Jaw, Eye
from .smplh import SMPLHinit, SMPLH

@dataclass 
class SMPLXinit(SMPLHinit):
    ext: str = 'npz'
    hand_vertex_ids_path: str = ''
    expression: Expression = Expression()
    jaw_pose: Jaw = Jaw()
    leye_pose: Eye = Eye()
    reye_pose: Eye = Eye()

@dataclass
class SMPLX(SMPLH):
    num_vertices: int = 10498
    init: SMPLXinit = SMPLXinit()