import torch 
import torch.nn as nn 
import numpy as np
import math
from typing import Optional, Dict, Union
import PIL.Image as pil_img
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    PointLights, PerspectiveCameras,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex

from .texturer import Texturer


class Pytorch3dRenderer(nn.Module):

    def __init__(
        self,
        cameras=None,
        image_height=224,
        image_width=224,
        blur_radius: float = 0.0, 
        faces_per_pixel: int = 5, # rasterizer settings
        light_location=[[0.0, 0.0, -0.5]],
    ):
        super(Pytorch3dRenderer, self).__init__()
        """
        Camera class that integrates a Pytorch3D Differentiable Renderer.
        We use the PyTorch3D coordinate system, which assumes: 
        +X:left, +Y: up and +Z: from us to scene (right-handed)
        See here: https://pytorch3d.org/docs/cameras 

        Renderes images for a given camera
        Parameters:
            camera_index: int
                The index of the camera to use.
        """

        # build renderer (while the camera class batch size can be arbitrary,
        # the renderer batch size is fixed to 1)
        self.cameras = cameras

        # get some parameters from camera class
        self.device = cameras.device
        self.iw = image_width
        self.ih = image_height
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel
        self.light_location = light_location


        self.build_renderer()
        self.texturer = Texturer() # give meshes a color

    def build_renderer(self):

        # lights 
        self.lights = PointLights(device=self.device, location=self.light_location)

        # rasterizer
        self.rasterizer = RasterizationSettings(
            image_size=(int(self.ih), int(self.iw)),
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel,
            max_faces_per_bin=100000,
        )

        # renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.rasterizer),
            shader=SoftPhongShader(cameras=self.cameras, lights=self.lights, device=self.device)
        )


    def vertices_faces_to_mesh(self, vertices, faces, textures):
        
        
        mesh_bs = vertices.shape[0]
        verts_list = [v for v in vertices]
        faces_list = mesh_bs * [faces]

        meshes = Meshes(
            verts=verts_list,
            faces=faces_list,
            textures=textures
        ) 
        meshes = join_meshes_as_scene(meshes)

        return meshes

    def build_meshes(self, 
        vertices, 
        faces, 
        textures=None, 
        body_model='smplx',
        colors = ['light_blue']
    ):
        '''Build a mesh object. '''

        mesh_bs = vertices.shape[0]

        if len(colors) == 1:
            colors = colors * mesh_bs

        if textures is None:
            textures = self.texturer.quick_texture(
                batch_size=mesh_bs, 
                body_model=body_model, 
                colors=colors
            ).to(self.device)

        meshes = self.vertices_faces_to_mesh(
            vertices, faces, textures
        )

        return meshes

    def create_floor(self, floor_transl=[0,0,0], size=5):

        floor_vertices = torch.tensor([
            [0, 0, 0],
            [0, 0, size],
            [size, 0, 0],
            [size, 0, size],
        ]).to(self.device)
        floor_faces = torch.tensor([[0, 1, 2], [1, 2, 3]]).to(self.device)

        # translate floor
        floor_vertices = floor_vertices - torch.tensor(floor_transl).to(self.device)

        # color the floor
        floor_rgb = 0.7 * torch.ones((1, 4, 3)).to(self.device)
        floor_textures = TexturesVertex(verts_features=floor_rgb)

        floor_mesh = Meshes(
                verts=[floor_vertices],
                faces=[floor_faces],
                textures=floor_textures
        )

        return floor_mesh


    def build_meshes_with_floor(self,
        vertices, 
        faces, 
        textures=None, 
        body_model='smplx',
        colors = ['light_blue'],
        floor_transl = [0,0,0]
    ):
        '''Build a mesh object. '''

        mesh_bs = vertices.shape[0]

        if len(colors) == 1:
            colors = colors * mesh_bs

        meshes = []
        for i in range(mesh_bs):
            textures = self.texturer.quick_texture(
                batch_size=1, 
                body_model=body_model, 
                colors=[colors[i]]
            ).to(self.device)

            meshes.append(Meshes(
                verts=[vertices[i]],
                faces=[faces],
                textures=textures
            ))
        
        # add floor 
        floor_vertices = torch.tensor([
            [5, 0, 5],
            [-5, 0, 5],
            [5, 0, -5],
            [-5, 0, -5],
        ]).to(self.device)
        floor_vertices = floor_vertices - torch.tensor(floor_transl).to(self.device)

        floor_faces = torch.tensor([[0, 1, 2], [1, 2, 3]]).to(self.device)
        floor_rgb = 0.7 * torch.ones((1, 4, 3)).to(self.device)
        floor_textures = TexturesVertex(verts_features=floor_rgb)
        meshes.append(Meshes(
                verts=[floor_vertices],
                faces=[floor_faces],
                textures=floor_textures
        ))

        meshes = join_meshes_as_scene(meshes)

        return meshes


    def render(self, 
        vertices, 
        faces, 
        out_fn=None, 
        textures=None, 
        body_model='smplx', 
        colors=['light_blue'],
        with_floor=False,
        floor_transl=[0,0,0]
    ):

        if with_floor:
            mesh = self.build_meshes_with_floor(
                vertices, faces, textures, body_model, colors, floor_transl
            )
        else:
            mesh = self.build_meshes(
                vertices, faces, 
                textures=textures, 
                body_model=body_model,
                colors=colors
            )

        image = self.renderer(mesh)
        
        if out_fn is not None:
            color_image = self.to_color(image)
            color_image.save(out_fn) 
        
        return image 
    
    def to_color(self, image):
        image = image.detach().cpu().numpy()
        if len(image.shape) == 4:
            image = image[0]
        color_image = pil_img.fromarray(
            (image * 255).astype(np.uint8)[...,:3]
        )
        return color_image

    def update_camera_pose(self,
        pitch, yaw, roll, tx, ty, tz,
        update_light_location=True,
    ):
        '''Render a mesh with a given camera pose. '''
        rotation = torch.tensor([[pitch, yaw, roll]]).to(self.device)
        rotation = euler_angles_to_matrix(rotation * math.pi / 180, "XYZ")
        translation = torch.tensor([[tx, ty, tz]]).to(self.device)

        # update camera pose
        self.cameras.R = rotation
        self.cameras.T = translation

        # get new light location from new camera pose
        if update_light_location:
            new_light_location = self.cameras.get_camera_center() 
            self.renderer.shader.lights.location = new_light_location


if __name__ == "__main__":
    camren = MasterRenderer()
    import ipdb;ipdb.set_trace()