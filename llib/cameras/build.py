import torch
from llib.cameras.perspective import PerspectiveCamera


def build_camera(
    camera_cfg, 
    camera_type=None, 
    batch_size=1, 
    device='cuda'
):
    """
    Build camera.
    Parameters
    ----------
    camera_cfg: cfg
        config file of camera
    camera_type: str, optional
        type of camera (perspective)
    batch_size: int, optional
        batch size
    device: str, optional
        device to use
    """

    if camera_type == 'perspective':

        # unpack config params
        cfg = camera_cfg.perspective
        rotation = torch.tensor([[cfg.pitch, cfg.yaw, cfg.roll]])
        translation = torch.tensor([[cfg.tx, cfg.ty, cfg.tz]])
        afov_horizontal = torch.tensor([cfg.afov_horizontal])
        image_size = torch.tensor([[cfg.iw, cfg.ih]])

        camera = PerspectiveCamera(
            rotation=rotation,
            translation=translation,
            afov_horizontal=afov_horizontal,
            image_size=image_size,
            batch_size=batch_size,
            device=device
        )
    else:
        raise NotImplementedError

    return camera 
