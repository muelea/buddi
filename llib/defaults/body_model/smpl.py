import os
import os.path as osp
from dataclasses import dataclass
from typing import Optional
from .smpl_family_params import Shape, BodyPose, GlobalOrient, Translation
from .joint_mapper import JointMapper

@dataclass 
class SMPLinit:
    ext: str = 'pkl'
    batch_size: int = 1
    gender: str = 'neutral'
    age: str = 'adult'
    kid_template_path: str = 'essentials/body_models/smil/smplx_kid_template.npy'
    betas: Shape = Shape()
    body_pose: BodyPose = BodyPose()
    global_orient: GlobalOrient = GlobalOrient()
    transl: Translation = Translation() 
    joint_mapper: JointMapper = JointMapper()
    data_struct = None
    vertex_ids = None
    v_template = None

@dataclass
class SMPL:
    num_vertices: int = 6890
    init: SMPLinit = SMPLinit()