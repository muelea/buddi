import torch 
import torch.nn as nn
import numpy as np
from typing import NewType, List, Union, Tuple, Optional
from .alignment import build_alignment 

class PointError(nn.Module):
    def __init__(
        self,
        name,
        alignment,
    ):
        super(PointError, self).__init__()

        self.align = build_alignment(alignment)

    def forward(self, input_points, target_points):
        input_points, target_points = self.align(input_points, target_points)
        error = point_error(input_points, target_points)
        return error

def point_error(input_points, target_points):
    ''' Calculate point error
    Parameters
    ----------
        input_points: numpy.array, BxPx3
            The estimated points
        target_points: numpy.array, BxPx3
            The ground truth points
    Returns
    -------
        numpy.array, BxJ
            The point error for each element in the batch
    '''
    if torch.is_tensor(input_points):
        input_points = input_points.detach().cpu().numpy()
    if torch.is_tensor(target_points):
        target_points = target_points.detach().cpu().numpy()

    return np.sqrt(np.power(input_points - target_points, 2).sum(axis=-1))

def mpjpe(input_points, target_points):
    return point_error(input_points, target_points)

def v2v(input_points, target_points):
    return point_error(input_points, target_points)