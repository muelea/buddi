import numpy as np
import torch

def compute_rotation_matrix(pitch, yaw, roll):
    # compute rotation matrix
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180
    roll = roll * np.pi / 180
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    R = torch.tensor(
        [[[cos_y*cos_r, sin_p*sin_y*cos_r - cos_p*sin_r, cos_p*sin_y*cos_r + sin_p*sin_r],
        [cos_y*sin_r, sin_p*sin_y*sin_r + cos_p*cos_r, cos_p*sin_y*sin_r - sin_p*cos_r],
        [-sin_y, sin_p*cos_y, cos_p*cos_y]]]
    )
    return R[0]

def renderer_lookat(
        XYZ=[1.17508748, 0.50318876, 3.24840362],
        xyz=[400, 400, 1],
        cxcy=[400, 400],
        focal_length_px=1200,
        tz_dist=5.0,
        pitch=0, 
        roll=0, 
        yaw=0,
):

    R = compute_rotation_matrix(pitch, yaw, roll)

    # World point to camera coordinates
    RXYZ = torch.matmul(R, torch.tensor(XYZ).double())

    tz = 1 - RXYZ[2] + tz_dist
    tx = (((RXYZ[2] + tz) * (xyz[0] - cxcy[0])) / focal_length_px) - RXYZ[0]
    ty = (((RXYZ[2] + tz) * (xyz[1] - cxcy[1])) / focal_length_px) - RXYZ[1]

    camera_params = [pitch, yaw, roll, tx.item(), ty.item(), tz.item()]

    return camera_params