import os
import pickle
import numpy as np
import torch
import smplx
import imageio

from llib.cameras.perspective import PerspectiveCamera
from llib.visualization.pyrenderer import PyRenderer


def render_two_meshes(renderer, vertices, faces):
    """
    :param renderer
    :param vertices (B, V, 3)
    :param faces (F, 3)
    """
    vertex_transl_center = vertices.mean((0, 1))
    verts_centered = vertices - vertex_transl_center
    renderer.update_meshes(verts_centered, faces)

    renderer.update_camera_euler(0.0, 0.0, 0.0, 0.0, 0.0, 2.0)
    side_view = renderer.render()

    renderer.update_camera_euler(270, 0.0, 0.0, 0.0, 2.0, 0.0)
    top_view = renderer.render()

    return side_view, top_view


if __name__ == "__main__":

    batch_size = 1
    body_model_type = "smplx"
    afov_horizontal = 60
    pitch, roll, yaw = 0.0, 0.0, 0.0
    tx, ty, tz = 0.0, 0.0, 0.0
    iw, ih = 200, 256

    # create renderer to visualize results
    rotation = torch.tensor([[pitch, yaw, roll]])
    translation = torch.tensor([[tx, ty, tz]])
    afov_horizontal = torch.tensor([afov_horizontal])
    image_size = torch.tensor([[iw, ih]])

    camera = PerspectiveCamera(
        rotation=rotation,
        translation=translation,
        afov_horizontal=afov_horizontal,
        image_size=image_size,
        batch_size=batch_size,
    )

    renderer = PyRenderer(cameras=camera.cameras, image_width=iw, image_height=ih,)

    # create body model
    body_model = smplx.create(
        model_path="essentials/body_models/",
        gender="neutral",
        model_type=body_model_type,
    )
    faces = torch.tensor(body_model.faces.astype(np.int32))

    if True:
        path = "/shared/lmueller/essentials/example_data/diffusion/denoised_smplx.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        T = sorted(data.keys())[0]
        verts = torch.stack([data[T][0].vertices[0], data[T][1].vertices[0]], dim=0)
    else:
        body = body_model()
        verts = body.vertices.detach()
        verts = verts.repeat(2, 1, 1)
        verts[1, :, :] += 0.1

    side, top = render_two_meshes(renderer, verts, faces)
    imageio.imwrite("side.png", side)
    imageio.imwrite("top.png", top)
