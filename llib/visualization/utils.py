import numpy as np 


def add_alpha_channel(image, alpha=255.0):
    """Add alpha channel to image. """

    alpha_channel = alpha * np.ones_like(image[...,[0]])
    image = np.concatenate([image, alpha_channel], axis=-1)
    return image

def overlay_images(background, overlay, alpha_treshold=0.0):
    """Overlay background image with overlay image for overlay alpha
    values larger than threshold. """
    if background.shape[-1] == 3:
        background = add_alpha_channel(background, alpha=1.0)
    
    if overlay.shape[-1] != 4:
        raise ValueError('Overlay image must have alpha channel')

    mask = overlay[..., -1] > alpha_treshold
    background[mask] = overlay[mask]

    return background