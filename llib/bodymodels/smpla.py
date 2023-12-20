from smplx import SMPLX
import torch
import torch.nn as nn
from typing import Optional
from collections import namedtuple

TensorOutput = namedtuple('TensorOutput',
                          ['vertices', 'joints', 'betas', 'scale',
                          'expression', 
                          'global_orient', 'body_pose', 'left_hand_pose',
                           'right_hand_pose', 'jaw_pose', 'transl', 'full_pose',
                           'v_shaped'])

class SMPLXA(SMPLX):
    def __init__(self,
        **kwargs
    ):
        """ 
        SMPL-XA Model, which extends SMPL-X to children and adults.
        Parameters
        ----------
        kwargs:
            Same as SMPL-X   
        """
        super(SMPLXA, self).__init__(**kwargs)

        default_scale = torch.zeros(
                    [self.batch_size, 1], dtype=self.dtype)
        self.register_parameter(
                'scale', nn.Parameter(default_scale, requires_grad=True))

        default_betas = torch.zeros(
                    [self.batch_size, self.num_betas - 1], dtype=self.dtype)
        self.register_parameter(
            'betas', nn.Parameter(default_betas, requires_grad=True))


    def name(self) -> str:
        return 'SMPL-XA'

    def forward(
        self,
        betas: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        body_pose: Optional[torch.Tensor] = None,
        **kwargs
    ):

        betas = betas if betas is not None else self.betas
        scale = scale if scale is not None else self.scale
        betas_smpla = torch.cat([betas, scale], dim=1)

        transl = transl if transl is not None else self.transl

        body_pose = body_pose if body_pose is not None else self.body_pose
        
        global_orient = global_orient if global_orient is not None else self.global_orient

        body = super(SMPLXA, self).forward(
            betas=betas_smpla, 
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose,
            **kwargs
        )

        output = TensorOutput(vertices=body.vertices,
                             joints=body.joints,
                             betas=betas_smpla[:,:-1],
                             scale=scale,
                             expression=body.expression,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=body.left_hand_pose,
                             right_hand_pose=body.right_hand_pose,
                             jaw_pose=body.jaw_pose,
                             v_shaped=body.v_shaped,
                             transl=transl,
                             full_pose=body.full_pose)

        return output

