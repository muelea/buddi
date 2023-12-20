import os
import numpy as np
import torch

os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender
from pyrender.constants import RenderFlags
from pyrender.light import DirectionalLight
from pyrender.node import Node
import trimesh


class PyRenderer(object):
    """
    Pyrender renderer to be interfaced with existing pytorch3d renderer
    Uses coordinates with X right Y up Z back
    :param cameras (pytorch3d camera)
    :param image_height (int default 224)
    :param image_width (int default 224)
    :param light_location (list of (3,) or tensor (N, 3))
    """

    def __init__(
        self,
        cameras,
        image_height=224,
        image_width=224,
        **kwargs,
    ):
        self.scene = pyrender.Scene(
            ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0, 0.0]
        )
        self.scene.bg_color = np.array([1.0, 1.0, 1.0, 0.0])

        self.viewport_size = (image_width, image_height)
        self.viewer = pyrender.OffscreenRenderer(*self.viewport_size)

        self.cameras = cameras_p3d_to_pyrender(cameras)
        self.camera_nodes = [
            self.scene.add(cam, name=f"camera-{i}")
            for i, cam in enumerate(self.cameras)
        ]
        self.scene.main_camera_node = self.camera_nodes[0]
        self.mesh_nodes = []

        self.add_ground()
        self.add_lighting()

    def get_intrinsics(self):
        iw, ih = self.viewer.viewport_width, self.viewer.viewport_height
        cam = self.scene.main_camera_node.camera
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        return fx, fy, cx, cy, iw, ih

    def update_intrinsics(self, fx, fy, cx, cy, iw, ih):
        self.viewport_size = (iw, ih)
        self.viewer.viewport_width = iw
        self.viewer.viewport_height = ih

        cam = self.scene.main_camera_node.camera
        cam.fx, cam.fy, cam.cx, cam.cy = fx, fy, cx, cy

    def add_ground(self):
        ground = pyrender.Mesh.from_trimesh(
            make_checkerboard(up="y", color0=[0.95,0.95,0.95], color1=[0.95,0.95,0.95], alpha=1.0), smooth=False
        )
        self.ground_node = self.scene.add(ground, name="ground")
        pose = make_4x4_pose(torch.eye(3), torch.tensor([0.0, 10.0, 0.0])).numpy()
        self.update_ground_pose(pose)

    def update_ground_pose(self, pose):
        self.ground_pose = np.asarray(pose).astype(float)
        self.scene.set_pose(self.ground_node, self.ground_pose)

    def add_lighting(self, n_lights=6, elevation=np.pi / 6, dist=4):
        light_poses = get_light_poses(n_lights, elevation, dist)
        self.light_nodes = []
        for i, pose in enumerate(light_poses):
            node = Node(
                name=f"light-{i:02d}",
                light=DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=pose,
            )
            if self.scene.has_node(node):
                continue
            self.scene.add_node(node)
            self.light_nodes.append(node)

    def set_mesh_visibility(self, vis):
        for node in self.mesh_nodes:
            node.mesh.is_visible = vis

    def update_meshes(self, verts, faces, colors=None, smooth=True):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (optional tensor, (B, 3))
        """
        # set the ground origin to be at the min point
        verts_min = verts.amax(dim=(0, 1))  # (3)
        ground_pose = transform_pyrender(make_4x4_pose(torch.eye(3), verts_min)).numpy()
        self.update_ground_pose(ground_pose)

        if colors is None:
            colors = get_colors()[: len(verts)]  # (B, 3)
        meshes_t = make_batch_trimesh(verts, faces, colors)
        meshes = [pyrender.Mesh.from_trimesh(m, smooth=smooth) for m in meshes_t]

        # re-use existing mesh nodes
        self.set_mesh_visibility(False)
        for i, mesh in enumerate(meshes):
            if i < len(self.mesh_nodes):
                self.mesh_nodes[i].mesh = mesh
                self.mesh_nodes[i].mesh.is_visible = True
            else:
                self.mesh_nodes.append(self.scene.add(mesh, name=f"mesh_{i:03d}"))

    def update_camera_matrix(self, pose):
        """
        :param pose (4, 4) tensor camera to world transform
        """
        pose = np.asarray(pose)
        self.scene.set_pose(self.scene.main_camera_node, pose)

    def update_camera_euler(self, pitch, yaw, roll, tx, ty, tz):
        """
        Inputs are scalars specifying the camera to world transform
        """
        rot = torch.tensor([pitch, yaw, roll])
        rot = make_rotation(rot * np.pi / 180, "xyz")  # (3, 3)
        pose = make_4x4_pose(rot, torch.tensor([tx, ty, tz]))  # (4, 4)
        self.update_camera_matrix(pose)

    def render(self, show_ground=False, **kwargs):
        """
        :param out_fn (optional callable for PIL image, default None)
        """
        self.ground_node.mesh.is_visible = show_ground
        flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL
        rgba, _ = self.viewer.render(self.scene, flags=flags)
        return rgba


def get_colors():
    file = os.path.abspath(os.path.join(__file__, "../colors.txt"))
    return torch.from_numpy(np.loadtxt(file))


def make_batch_trimesh(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (F, 3)
    :param colors (B, 3)
    """
    B, V = verts.shape[:2]
    return [
        make_trimesh(verts[b], faces, colors[b, None].expand(V, -1)) for b in range(B)
    ]


def make_trimesh(verts, faces, color=None, yup=True):
    """
    :param verts torch tensor (V, 3)
    :param faces torch tensor (F, 3)
    :param color (optional torch tensor (V, 3))
    """
    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    if yup:
        verts = np.array([1, -1, -1])[None, :] * verts
    if color is None:
        color = np.ones_like(verts) * 0.5
    else:
        color = color.detach().cpu().numpy()
    return trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_colors=color, process=False
    )


def cameras_p3d_to_pyrender(cameras):
    """
    Convert pytorch3d cameras to pyrender cameras
    """
    cameras = cameras.to("cpu")
    K = cameras.K  # (B, 4, 4)
    # intrins (B, 4) (fx, fy, cx, cy)
    intrins = torch.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], dim=-1)
    B = K.shape[0]
    cameras_pr = []
    for b in range(B):
        fx, fy, cx, cy = intrins[b]
        cameras_pr.append(pyrender.IntrinsicsCamera(fx, fy, cx, cy))

    return cameras_pr


def transform_pyrender(T_c2w):
    """
    :param T_c2w (*, 4, 4)
    """
    T_vis = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=T_c2w.device,
    )
    return torch.einsum(
        "...ij,jk->...ik", torch.einsum("ij,...jk->...ik", T_vis, T_c2w), T_vis
    )


def pose_p3d_to_pyrender(pose):
    T = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return torch.einsum("ij,...jk->...ik", T, pose)


def get_light_poses(n_lights=5, elevation=np.pi / 6, dist=6.0):
    # get lights in a circle around origin at elevation
    thetas = elevation * torch.ones(n_lights)
    phis = 2 * np.pi * torch.arange(n_lights) / n_lights
    euler_angles = torch.stack([-thetas, phis, torch.zeros(n_lights)], dim=-1)  # (N, 3)
    rots = make_rotation(euler_angles, order="xyz")  # (N, 3, 3)
    trans = torch.einsum("nij,j->ni", rots, torch.tensor([0.0, 0.0, dist]))  # (N, 3)
    return make_4x4_pose(rots, trans).numpy()


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    """
    dims = R.shape[:-2]
    pose = torch.eye(4).reshape(*(1,) * len(dims), 4, 4).repeat(*dims, 1, 1)
    pose[..., :3, :3] = R
    pose[..., :3, 3] = t
    return pose

def make_1x1_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    """
    dims = R.shape[:-2]
    pose = torch.eye(4).reshape(*(1,) * len(dims), 1, 1).repeat(*dims, 1, 1)
    pose[..., :3, :3] = R
    pose[..., :3, 3] = t
    return pose

def make_rotation(euler_angles, order="xyz"):
    order = order.lower()
    thetas = euler_angles.unbind(dim=-1)
    mats = [get_rot_mat(theta, axis) for theta, axis in zip(thetas, order)]
    return torch.matmul(mats[2], torch.matmul(mats[1], mats[0]))


def get_rot_mat(theta, axis):
    axis = axis.lower()
    one = torch.ones_like(theta)
    zero = torch.zeros_like(theta)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    if axis == "x":
        mat_flat = [one, zero, zero, zero, cos, -sin, zero, sin, cos]

    elif axis == "y":
        mat_flat = [cos, zero, sin, zero, one, zero, -sin, zero, cos]

    elif axis == "z":
        mat_flat = [cos, -sin, zero, sin, cos, zero, zero, zero, one]

    else:
        raise ValueError

    return torch.stack(mat_flat, dim=-1).reshape(*theta.shape, 3, 3)


def make_checkerboard(
    length=10.0,
    color0=[0.95, 0.95, 0.95],
    color1=[0.75, 0.75, 0.75],
    tile_width=0.5,
    alpha=1.0,
    up="y",
):
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    radius = length / 2.0
    num_rows = num_cols = int(length / tile_width)
    verts = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)  # (4, 3)
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)  # (4, 3)

            cur_faces = np.array(
                [[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=int
            )
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            verts.append(cur_verts)
            faces.append(cur_faces)
            face_colors.append(cur_colors)

    verts = np.concatenate(verts, axis=0).astype(float)
    faces = np.concatenate(faces, axis=0).astype(float)
    face_colors = np.concatenate(face_colors, axis=0).astype(float)

    return trimesh.Trimesh(verts, faces, face_colors=face_colors, process=False)
