import torch
from torch.nn import functional as F
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)


def axis_angle_to_rotation6d(x):
    rotmat = axis_angle_to_matrix(x)
    rot6d = matrix_to_rotation_6d(rotmat)
    return rot6d


def rotation6d_to_axis_angle(x):
    rotmat = rotation_6d_to_matrix(x)
    rot_aa = matrix_to_axis_angle(rotmat)
    return rot_aa


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [*, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [*, 3, 3]
    """
    angle = torch.linalg.norm(theta + 1e-5, dim=-1, keepdim=True)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=-1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [*, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [*, 3, 3]
    """
    norm_quat = quat / torch.linalg.norm(quat + 1e-5, dim=-1, keepdim=True)
    w, x, y, z = norm_quat.unbind(dim=-1)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=-1,
    ).view(*quat.shape[:-1], 3, 3)
    return rotMat
