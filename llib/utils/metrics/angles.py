import torch 
import math
from pytorch3d.transforms import axis_angle_to_matrix


def angle_error(a1, a2, eps=1e-10):
    """ Calculate angle error between two axis-angle rotations
    Parameters
    ----------
    a1 : torch.Tensor
        axis-angle rotation of shape (..., 3)
    a2 : torch.Tensor
        axis-angle rotation of shape (..., 3)
    eps:
        epsilon for numerical stability

    Returns
    -------
    torch.Tensor
        angle error in degrees of shape (...)
    
    """
    # to rotation matrix 
    m1 = axis_angle_to_matrix(a1)
    m2 = axis_angle_to_matrix(a2)

    # flatten to vector 
    vec_shape = [x for x in m1.shape[:-2]]
    m1 = m1.view(vec_shape + [-1])
    m2 = m2.view(vec_shape + [-1])

    # get angle error
    dotprod = torch.sum(m1 * m2, dim=-1)
    denom = torch.norm(m1, dim=-1) * torch.norm(m2, dim=-1) + eps
    angle = dotprod / denom

    return torch.acos(angle) * 180 / math.pi