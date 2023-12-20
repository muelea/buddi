import numpy as np
import torch
from tqdm import tqdm
from llib.cameras.perspective import PerspectiveCamera
from llib.visualization.pyrenderer import PyRenderer, make_rotation, make_4x4_pose


def build_camera(width, height, afov_horizontal):
    afov_horizontal = torch.tensor([afov_horizontal])
    image_size = torch.tensor([[width, height]])

    return PerspectiveCamera(
        afov_horizontal=afov_horizontal, image_size=image_size, batch_size=1,
    )


def afov_to_focal_length(afov, width, height):
    """
    Convert angular field of view to focal length in pixels.
    length_px is the height or width of the image in pixels.
    """
    length = max(width, height)
    afov = (afov / 2) * np.pi / 180
    focal_length = (length / 2) / np.tan(afov)
    return focal_length


def build_renderer(cfg=None):
    # create renderer to visualize results
    afov_horizontal = cfg.afov_horizontal if cfg is not None else 60
    width = cfg.width if cfg is not None else 200
    height = cfg.height if cfg is not None else 256

    camera = build_camera(width, height, afov_horizontal)

    return PyRenderer(cameras=camera.cameras, image_width=width, image_height=height,)


def move_to(tensor, device="cpu"):
    return torch.as_tensor(tensor).to(device)


def composite_rgba(rgba, bg):
    rgba, bg = rgba / 255, bg / 255
    alpha = rgba[..., 3:]
    comp = alpha * rgba[..., :3] + (1 - alpha) * bg
    return (255 * comp).astype(np.uint8)


def get_camera_intrins(width, height, afov_horizontal):
    cx, cy = 0.5 * width, 0.5 * height
    focal = afov_to_focal_length(afov_horizontal, width, height)
    return focal, focal, cx, cy


def render_camera_view(renderer, verts, faces, bg_img=None, **kwargs):
    renderer.update_meshes(verts, faces)
    renderer.update_camera_matrix(np.eye(4))
    frame = renderer.render(**kwargs)
    if bg_img is not None:
        frame = composite_rgba(frame, bg_img)
    return frame


def render_top_view(renderer, verts, faces, angle=90, **kwargs):
    verts = verts - verts.mean((0, 1))
    renderer.update_meshes(verts, faces)

    euler_angles = torch.tensor([-angle * np.pi / 180, 0, 0])
    rot = make_rotation(euler_angles)  # (3, 3)
    trans = rot @ torch.tensor([0.0, 0.0, 2.0])  # (3,)
    top_pose = make_4x4_pose(rot, trans)
    renderer.update_camera_matrix(top_pose.numpy())

    return renderer.render(**kwargs)


def get_spiral_poses(num_poses, elev_min=0, elev_max=0, num_reps=2, dist=2, **kwargs):
    # let the pitch oscillate (sinusoid) between elev_min and elev_max num_reps times
    pitch = torch.sin(torch.linspace(0, num_reps * 2 * np.pi, num_poses))
    pitch = 0.5 * (elev_max - elev_min) * pitch + 0.5 * (elev_max + elev_min)
    yaw = torch.linspace(0, 2 * np.pi, num_poses)
    roll = torch.zeros(num_poses)
    euler_angles = torch.stack([pitch, yaw, roll], dim=-1)
    rots = make_rotation(euler_angles, order="xyz")  # (N, 3, 3)
    trans = torch.einsum("nij,j->ni", rots, torch.tensor([0.0, 0.0, dist]))  # (N, 3)
    return make_4x4_pose(rots, trans)


def render_360_views(renderer, verts, faces, num_poses=30, **kwargs):
    verts = verts - verts.mean((0, 1))
    renderer.update_meshes(verts, faces)

    cam_poses = get_spiral_poses(num_poses, **kwargs)

    frames = []
    for cam in tqdm(cam_poses):
        renderer.update_camera_matrix(cam)
        frames.append(renderer.render(**kwargs))
    return frames


def align_pcl(Y, X, weight=None, fixed_scale=False):
    """align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    X = torch.as_tensor(X).clone()
    Y = torch.as_tensor(Y).clone()
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = (
            torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(
                dim=-1, keepdim=True
            )
            / var[..., 0]
        )  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t

def crop_image_from_alpha(image, alpha=None, to_square=True, pad_factor=0.1):
    """
    Crop image to the bounding box of the non-zero alpha channel.
    The cropped image is padded by pad_factor * bbox_size.
    If to_square the bounding box is squared to max(width, height).
    Params:
    -------
    image: np.array of shape (H, W, 4) or (H, W, 3) if alpha is provided
    alpha: np.array of shape (H, W, 1) or None
    to_square: bool, whether to square the bounding box. This might not be
        possible when square bbox exceeds image bounds.
    pad_factor: float, amount to pad the bounding box by
 
    Returns:
    --------
    cropped_image: np.array of shape (H', W', 3)
    bbox: np.array of shape (4,) containing the bounding box
    """

    iw, ih, channels = image.shape

    if alpha is None:
        assert image.shape[-1] == 4, "image must have 4 channels"
        alpha = image[...,-1]
    
    # get bounding box of non-zero alpha channel
    mask = alpha > 0
    rmin, rmax = np.where(np.any(mask, axis=1))[0][[0, -1]]
    cmin, cmax = np.where(np.any(mask, axis=0))[0][[0, -1]]

    # expand bbox by pad_factor
    if pad_factor != 0 or to_square:
        bbox_size = [rmax - rmin, cmax - cmin]
        if to_square:
            bbox_size = [max(bbox_size), max(bbox_size)]

        if pad_factor != 0:
            bbox_size = [int(x + pad_factor * x) for x in bbox_size]

        # check if bbox exceeds image bounds, if so, use image bounds
        rcen = (rmin + rmax) // 2
        ccen = (cmin + cmax) // 2
        rmin, rmax = rcen - bbox_size[0] // 2, rcen + bbox_size[0] // 2
        cmin, cmax = ccen - bbox_size[1] // 2, ccen + bbox_size[1] // 2

        # crop image or pad when bbox exceeds image bounds
        rmin, rmax = max(rmin, 0), min(rmax, iw)
        cmin, cmax = max(cmin, 0), min(cmax, ih)

    # output bounding box and and crop image
    bbox = {'rmin': rmin, 'rmax': rmax, 'cmin': cmin, 'cmax': cmax}
    image = image[rmin:rmax, cmin:cmax]

    # pad image if to square
    if to_square:
        fbs = max(image.shape[:2]) # final bounding box size
        pad_image = 255 * np.ones([fbs, fbs, channels], dtype=image.dtype)
        pad_image[:rmax-rmin, :cmax-cmin] = image
        image = pad_image

    return image, bbox