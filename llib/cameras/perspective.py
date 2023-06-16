import torch 
import torch.nn as nn 
import math
from typing import Optional
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix



class PerspectiveCamera(nn.Module):

    DEFAULT_AFOV = 60
    DEFAULT_IMAGE_SIZE = 224

    def __init__(
        self,
        rotation: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None, 
        afov_horizontal: Optional[torch.Tensor] = None, 
        image_size: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: str = "cuda",
    ):
        super().__init__()
        """
        Camera class that integrates a with Pytorch3D Differentiable Renderer.
        We use the PyTorch3D coordinate system, which assumes: 
        +X:left, +Y: up and +Z: from us to scene (right-handed)
        See here: https://pytorch3d.org/docs/cameras 

        Parameters:
        ----------- 
        rotation: torch.Tensor
            Camera rotation in Euler angles (pitch, yaw, roll). Size be of size (N, 3) or (1, 3).
            If (1,3) the initial rotation is repeated for all N cameras.
        translation: torch.Tensor
            Camera translation (tx, ty, tz). Size be of size (N, 3) or (1, 3).
            If (1,3) the initial translation is repeated for all N cameras.
        afov_horizontal: torch.Tensor
            Horizontal angular field of view in degrees (uses image width). Size be of size (N, 1) or (1, 1).
            If (1,3) the initial afov_horizontal is repeated for all N cameras.
        image_size: torch.Tensor
            Image width, height. Size be of size (N, 1) or (1, 2).
            If (1,3) the initial image_size is repeated for all N cameras.
        device: torch.device
            Device to use for rendering. Cameras setup needs device when built.
        batch_size: int
            Batch size N. Should be equal to rotation, translation, afov_horizontal and image_size
            dimensions if those are specified.

        """

        self.batch_size = batch_size
        self.device = device

        # bev uses max(ih, iw) for compute focal length in pixel from AFOV 
        self.fl2px_max = False #True

        # get default camera params if not speficied and create member variable for each param
        self.parse_arguments(
            rotation, translation, afov_horizontal, image_size
        )

        self.cameras = self.build_cameras()

    def parse_arguments(self, rotation=None, translation=None, afov_horizontal=None, image_size=None):
        """
        Process input arguments. If None specified, use default values.
        """

        def repeat_param(param, param_name, batch_size):
            '''If a parameter is given, check if it matches the batch size and repeat if needed.'''

            if param.shape[0] == batch_size:
                pass
            elif param.shape[0] == 1: # if param batch size is N, repeat for all N cameras
                param = param.repeat(batch_size, 1)
            else:
                raise ValueError(
                    f'Camera Error: Batch size of {param_name} != 1 and != {batch_size} (camera batch_size).'
                )
            return param

        # camera rotation
        rotation = torch.zeros([self.batch_size, 3]) if rotation is None \
            else repeat_param(rotation, 'rotation', self.batch_size)
        self.init_pitch, self.init_yaw, self.init_roll = self.unpack_rotation(rotation)
        self.register_parameter('pitch', nn.Parameter(self.init_pitch))
        self.register_parameter('yaw', nn.Parameter(self.init_yaw))
        self.register_parameter('roll', nn.Parameter(self.init_roll))

        # camera translation
        translation = torch.zeros([self.batch_size, 3]) if translation is None \
            else repeat_param(translation, 'translation', self.batch_size)
        self.init_tx, self.init_ty, self.init_tz = self.unpack_translation(translation)
        self.register_parameter('tx', nn.Parameter(self.init_tx))
        self.register_parameter('ty', nn.Parameter(self.init_ty))
        self.register_parameter('tz', nn.Parameter(self.init_tz))

        # camera afov
        image_size = self.DEFAULT_IMAGE_SIZE * torch.ones([self.batch_size, 2]) if image_size is None \
            else repeat_param(image_size, 'image_size', self.batch_size)
        self.register_buffer('iw', image_size[:, [0]]) # image height
        self.register_buffer('ih', image_size[:, [1]]) # image width

        # focal length
        afov_horizontal = self.DEFAULT_AFOV * torch.ones([self.batch_size, 1]) if afov_horizontal is None \
            else repeat_param(afov_horizontal, 'afov_horizontal', self.batch_size)
        length_px = torch.hstack((self.iw, self.ih)).max(1)[0].unsqueeze(1) if self.fl2px_max else self.iw
        self.init_fl = self.afov_to_focal_length_px(afov_horizontal, length_px)
        self.register_parameter('fl', nn.Parameter(self.init_fl))


    def afov_to_focal_length_px(self, afov, length_px):
        '''
        Convert angular field of view to focal length in pixels.
        length_px is the height or width of the image in pixels.
        '''
        afov = (afov / 2) * math.pi / 180
        focal_length_px = (length_px/2) / torch.tan(afov)
        return focal_length_px

    def focal_length_px_to_afov(self, focal_length_px, length_px):
        '''
        Convert focal length in pixels to angular field of view.
        length_px is the height or width of the image in pixels.
        '''
        x = torch.atan(length_px / (2 * focal_length_px))
        afov = (2 * 180 * x) / math.pi
        return afov

    def unpack_rotation(self, rotation):
        pitch = rotation[:, [0]]
        yaw = rotation[:, [1]]
        roll = rotation[:, [2]]
        return pitch, yaw, roll

    def unpack_translation(self, translation):
        tx = translation[:, [0]]
        ty = translation[:, [1]]
        tz = translation[:, [2]]
        return tx, ty, tz
    
    def pack_translation(self, tx, ty, tz):
        return torch.cat([tx, ty, tz], dim=1)
    
    def pack_rotation(self, pitch, yaw, roll):
        return torch.cat([pitch, yaw, roll], dim=1)

    def ea2rm(self, pitch, yaw, roll)  -> torch.Tensor:
        """
        Convert Euler angles to rotation matrix.
        Axis rotation sequence is ZYX (yaw, pitch, roll).
        """

        # convert to radians
        pitch = pitch * math.pi / 180
        yaw = yaw * math.pi / 180
        roll = roll * math.pi / 180
        
        # create rotation matrix
        ea = self.pack_rotation(pitch, yaw, roll)
        R = euler_angles_to_matrix(ea, "XYZ")

        return R

    def get_calibration_matrix(self):
        # follow the convention of pytorch3d
        # https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.PerspectiveCameras.get_projection_transform
        principal_point = torch.hstack((self.iw/2, self.ih/2))
        calibration_matrix = torch.zeros((self.batch_size, 4, 4)).to(self.device)
        calibration_matrix[:, 0, 0] = self.fl[:, 0]
        calibration_matrix[:, 1, 1] = self.fl[:, 0]
        calibration_matrix[:, 0, 2] = principal_point[:, 0]
        calibration_matrix[:, 1, 2] = principal_point[:, 1]
        calibration_matrix[:, 3, 2] = 1
        calibration_matrix[:, 2, 3] = 1
        return calibration_matrix

    def build_cameras(self):

        # Initialize an OpenGL perspective camera.     
        camera_rotation = self.ea2rm(self.pitch, self.yaw, self.roll)
        camera_translation = self.pack_translation(self.tx, self.ty, self.tz)
        focal_length = self.fl  
        principal_point = torch.hstack((self.iw/2, self.ih/2))
        image_size = torch.hstack((self.ih, self.iw)).to(torch.int32)
        calibration_matrix = self.get_calibration_matrix()

        cameras = PerspectiveCameras(
            #principal_point=principal_point,
            #focal_length=focal_length,
            image_size=image_size,
            R=camera_rotation,
            T=camera_translation,
            K=calibration_matrix,
            device=self.device,
            in_ndc=False,
        )

        return cameras

    def project(self, points: torch.Tensor) -> torch.Tensor:
        
        # project points using pytorch3d camera
        self.cameras.R = self.ea2rm(self.pitch, self.yaw, self.roll)
        self.cameras.T = self.pack_translation(self.tx, self.ty, self.tz)
        self.cameras.image_size = torch.hstack((self.ih, self.iw)).to(torch.int32)
        #use K instead
        #self.cameras.focal_length = self.fl
        #self.cameras.principal_point = torch.hstack((self.iw/2, self.ih/2))
        self.cameras.K = self.get_calibration_matrix()

        projected_points = self.cameras.transform_points_screen(points)[:,:,:2]

        return projected_points

if __name__ == "__main__":
    camera = PerspectiveCamera()