from .renderer import Pytorch3dRenderer


def build_renderer(
    renderer_cfg,
    renderer_type,
    camera=None,
    batch_size=1, 
    device='cuda'
):
    """
    Read the main config file and create body model.
    Keys are: cfg.body_model, cfg.device, and cfg.batch_size
    If keys are not passed params are read from config
    """
    if renderer_type == 'pytorch3d':
        cfg = renderer_cfg.pytorch3d
        light_location = []
        for x in cfg.light_location:
            light_location.append([u for u in x])
        renderer = Pytorch3dRenderer(
            cameras = camera,
            image_width=renderer_cfg.image_width,
            image_height=renderer_cfg.image_height,
            blur_radius=cfg.blur_radius,
            faces_per_pixel=cfg.faces_per_pixel, 
            light_location=light_location, 
        )
    else:
        raise NotImplementedError

    return renderer 
