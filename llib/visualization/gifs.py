import torch
import numpy as np
from pytorch3d.renderer import TexturesVertex
from .renderer import Renderer
from .utils import *

def render_two_meshes_sequence(
        renderer: Renderer, 
        texture: TexturesVertex,
        verts1: list, 
        verts2: list, 
        faces: torch.tensor,
        device: str='cuda',
):
    """
    Renderes a sequence of two meshes together.
    """
    meshes_seq = []
    for v1, v2 in zip(verts1, verts2):
        v1v2 = prep_vvf_for_rendering(
            renderer=renderer, 
            texture=texture,
            verts1=v1, verts2=v2, 
            faces=faces, device=device, 
            center_meshes=False)
        meshes_seq.append(v1v2)

    images = [renderer.render(mesh, None) for mesh in meshes_seq]
    images = [prep_image_output(x) for x in images]
    
    return images


def render_two_meshes_360(
        renderer: Renderer, 
        texture: TexturesVertex,
        verts1, 
        verts2, 
        faces: torch.tensor,
        device: str='cuda',
        center_meshes: bool=True,
        pitch: int=210,
        roll: int=0,
        yaws: list=np.arange(0, 360, 10),
        translation: list=[0, 0, 4],
):
    """
    Render two meshes with 360 degree rotation.
    renderer: pytroch3d renderer
    verts1: tensor of shape 1 x N x 3
    verts2: tensor of shape 1 x N x 3
    faces: tensor of size F x 3 x 3
    center_meshes: move meshes to center
    pitch/roll/yaw: camera rotation in degree
    translation: camera translation 
        [translation_x, translation_y, translation_z]
    """
    # render meshes and save image

    
    meshes = prep_vvf_for_rendering(
        renderer=renderer, texture=texture,
        verts1=verts1, verts2=verts2, faces=faces,
        device=device, center_meshes=center_meshes
    )

    images = []
    for yy in yaws:
        renderer.update_camera(
            pitch=pitch, yaw=yy, roll=roll, 
            tx=translation[0], 
            ty=translation[1], 
            tz=translation[2], 
            light_location=None
        )

        # given the camera position, compute light location
        # ToDo: there is a bug when camera_location_x = 0
        new_light_location = renderer.cameras.get_camera_center() 
        renderer.update_lights(new_light_location)

        images.append(renderer.render(meshes, None))

    images = [prep_image_output(x) for x in images]

    # this resolves the rendering bug when camera_location_x = 0
    images = images[1:]
    
    return images

def position_camera(point, dist, pitch, roll):
    """
    Given a 3D point, a distance and pitch and roll, position
    camera such that the point is centered in the image rendering.
    """
    point = torch.tensor([1,1,1])
    dist = 1.0

    pitch = 210
    roll = 0

    

def orbit_around_point(
        renderer: Renderer, 
        texture: TexturesVertex,
        verts1, 
        verts2, 
        faces: torch.tensor,
        orbit_center: torch.tensor,
        device: str='cuda',
        pitch: int=210,
        roll: int=0,
        yaws: list=np.arange(0, 360, 10),
        camera_translation: list=[0, 0, 4],
):
    """
    Render two meshes with 360 degree rotation.
    renderer: pytroch3d renderer
    verts1: tensor of shape 1 x N x 3
    verts2: tensor of shape 1 x N x 3
    faces: tensor of size F x 3 x 3
    center_meshes: move meshes to center
    pitch/roll/yaw: camera rotation in degree
    translation: camera translation 
        [translation_x, translation_y, translation_z]
    """

    # bring meshes / orbit point to origin
    meshes = prep_vvf_for_rendering(
        renderer=renderer, texture=texture,
        verts1=verts1, verts2=verts2, faces=faces,
        device=device, center_meshes=True,
        center=orbit_center
    )

    images = []
    for yy in yaws:
        renderer.update_camera(
            pitch=pitch, yaw=yy, roll=roll, 
            tx=camera_translation[0], 
            ty=camera_translation[1], 
            tz=camera_translation[2], 
            light_location=None
        )

        # given the camera position, compute light location
        # ToDo: there is a bug when camera_location_x = 0
        new_light_location = renderer.cameras.get_camera_center() 
        renderer.update_lights(new_light_location)

        images.append(renderer.render(meshes, None))

    images = [prep_image_output(x) for x in images]

    # this resolves the rendering bug when camera_location_x = 0
    images = images[1:]
    
    return images

def sequence_with_360(
        verts1_seq, 
        verts2_seq, 
        shot_change_idx,
        renderer: Renderer, 
        renderer360: Renderer, 
        texture: TexturesVertex,
        faces: torch.tensor,
        device: str='cuda',
        camera_params: list = [210, 0, 180, 0, -1.6, 1.0] # relative to mean vertex
):
    pp, rr, yy, tx, ty, tz = camera_params
    camera_rot = torch.tensor(camera_params[:3]).to(device).double()
    camera_transl = torch.tensor(camera_params[3:]).to(device).double()

    # the yaw values of the 360 rotation should start st the initial yaw
    yaws = [x%360 for x in np.arange(yy, yy+360, 2)]

    # reset camera location
    verts1_seq = prep_vertices(verts1_seq)
    verts2_seq = prep_vertices(verts2_seq)
    
    # get the mean vertex position of the seqence per subject.
    # This will be the reference point for our cameraß
    verts1_mean = verts1_seq.mean(1).mean(0)
    verts2_mean = verts2_seq.mean(1).mean(0)
    verts_mean = 0.5 * (verts1_mean + verts2_mean)

    # To render the sequence, position camera in meshes center
    R = renderer.create_rotmat(pp, yy, rr).to(device).float()
    Rvm = torch.matmul(R, verts_mean)[0]
    tx_seq, ty_seq, tz_seq = -Rvm  + camera_transl
    renderer.update_camera(
        pitch=pp, yaw=yy, roll=rr, 
        tx=tx_seq, ty=ty_seq, tz=tz_seq, 
        light_location=None
    )

    # from the update camera, crete the light location
    new_light_location = renderer.cameras.get_camera_center() 
    renderer.update_lights(new_light_location)

    # render the sequnece 
    images_sequence = render_two_meshes_sequence(
        renderer=renderer,
        texture=texture,
        verts1=verts1_seq, 
        verts2=verts2_seq, 
        faces=faces, 
        device=device,
    )

    # now do the 360 degree rotation. We orbit around the vertex mean position
    # orbit_center == verts_mean
    orbit_camera_transl = (torch.matmul(
        renderer.cameras.R[0], verts_mean.reshape(3,1).double()
    ) + renderer.cameras.T.reshape(3,1)).reshape(3,1)

    images360 = orbit_around_point(
        renderer=renderer360,
        texture=texture,
        verts1=verts1_seq[shot_change_idx-1], 
        verts2=verts2_seq[shot_change_idx-1], 
        faces=faces, 
        device=device,
        pitch=pp,
        roll=rr,
        yaws=yaws,
        orbit_center=verts_mean.tolist(), 
        camera_translation=orbit_camera_transl 
    )

    # concatenate images in correct order
    images = images_sequence[:shot_change_idx] + images360 + images_sequence[shot_change_idx:]
    
    return images