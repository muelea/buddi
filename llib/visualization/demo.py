import numpy as np
import torch
import smplx
import cv2 

from llib.visualization.renderer import Pytorch3dRenderer
from llib.visualization.texturer import Texturer
from llib.cameras.perspective import PerspectiveCamera

def render_two_meshes(
        renderer, 
        body_model_type, 
        vertices, 
        meshcol=['light_blue1', 'light_blue6'],
        faces_tensor=None,
    ):

    vertex_transl_center = vertices.mean((0,1))
    verts_centered = vertices - vertex_transl_center
    if verts_centered.device != 'cuda':
        verts_centered = verts_centered.to('cuda')
        faces_tensor = faces_tensor.to('cuda')

    renderer.update_camera_pose(0.0, 20.0, 180.0, 0.0, 0.2, 2.0)
    rendered_img = renderer.render(
        verts_centered, faces_tensor, colors=meshcol, 
        body_model=body_model_type)
    side_view_color_image = rendered_img[0].detach().cpu().numpy() * 255


    # bird view
    renderer.update_camera_pose(270, 0.0, 180.0, 0.0, 0.0, 2.0)
    rendered_img = renderer.render(
        verts_centered, faces_tensor, colors=meshcol, 
        body_model=body_model_type)
    top_view_color_image = rendered_img[0].detach().cpu().numpy() * 255

    return side_view_color_image, top_view_color_image

if __name__ == "__main__":

    device='cuda'
    batch_size = 1
    body_model_type = 'smplx'
    afov_horizontal = 60
    pitch, roll, yaw = 0.0, 0.0, 0.0
    tx, ty, tz = 0.0, 0.0, 0.0
    iw, ih = 200, 256

    # create renderer to visualize results
    rotation = torch.tensor([[pitch, yaw, roll]])
    translation = torch.tensor([[tx, ty, tz]])
    afov_horizontal = torch.tensor([afov_horizontal])
    image_size = torch.tensor([[iw, ih]])
    
    # create body model
    body_model = smplx.create(
        model_path='essentials/body_models/',
        gender='neutral',
        model_type=body_model_type,
    )
    body = body_model()
    verts = body.vertices.detach()
    verts = verts.repeat(2,1,1)
    verts[1,:,:] += 0.1
    faces = torch.tensor(body_model.faces.astype(np.int32))

    # perspective camera
    camera = PerspectiveCamera(
        rotation=rotation,
        translation=translation,
        afov_horizontal=afov_horizontal,
        image_size=image_size,
        batch_size=batch_size,
        device=device
    )

    # pytroch3d renderer
    renderer = Pytorch3dRenderer(
        cameras = camera.cameras,
        image_width=iw,
        image_height=ih,
    )

    side_img, top_img= render_two_meshes(
        renderer, 
        body_model_type, 
        verts, 
        meshcol=['light_blue1', 'light_blue6'],
        faces_tensor=faces
    )

    # save images
    #cv2.imwrite('side_img.png', side_img)
    #cv2.imwrite('top_img.png', top_img)