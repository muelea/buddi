from dataclasses import dataclass
from typing import Optional
from .smpl_family_params import HandPose
from .smpl import SMPLinit, SMPL

@dataclass
class SMPLHinit(SMPLinit):
    left_hand_pose: HandPose = HandPose()
    right_hand_pose: HandPose = HandPose()
    use_compressed: bool = True

@dataclass
class SMPLH(SMPL):
    num_vertices: int = 6890
    init: SMPLHinit = SMPLHinit()