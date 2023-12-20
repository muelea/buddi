import numpy as np 
import trimesh
import cv2
import imageio



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


def read_images_from_disk(image_paths):
    """reads all images in image list from disk"""
    images = [cv2.imread(item) for item in image_paths]
    return images


def images_to_video(image_list, output_path, frame_rate=30, img_src=None):
    """Save mp4 video of list of images"""

    if isinstance(image_list[0], str):
        image_list = read_images_from_disk(image_list)

    # Get the dimensions of the first image in the list
    height, width, channels = image_list[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # write images to video write
    for frame in image_list:
        out.write(np.uint8(frame[:,:,:3]))
    
    # save result
    out.release()
    print(f"Video saved to {output_path}")

    # save as image overlay
    if img_src is not None:

        IMG = cv2.imread(img_src)

        # create new video writer
        fourccc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        out_new = cv2.VideoWriter(output_path.replace('.mp4', '_overlay.mp4'), fourccc, frame_rate, (width, height))

        # write images to video write
        for frame in image_list:
            frame_overlay = overlay_images(IMG, frame)
            out_new.write(np.uint8(frame_overlay[:,:,:3]))

        out_new.release()

def images_to_gif(image_list, output_path, frame_rate=30):
    """Save gif of list of images"""
    
    if isinstance(image_list, str):
        image_list = read_images_from_disk(image_list)
    
    image_list = [x.astype(np.uint8) for x in image_list]
    
    imageio.mimsave(output_path, image_list, fps=frame_rate)
    print(f"Gif saved to {output_path}")


def save_obj(smpl_body, faces, output_path):
    verts = smpl_body.vertices.detach().cpu().numpy()[0]
    mesh = trimesh.Trimesh(verts, faces)
    _ = mesh.export(output_path)
    print(f"Mesh exported {output_path}")
