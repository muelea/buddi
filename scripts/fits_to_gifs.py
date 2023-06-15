import os
import imageio
import numpy as np
from PIL import Image

# Specify the folder path, number of rows, and number of columns
folder_path = "../projectpage/render_highres_v02/FlickrCI3D_Signatures_Transformer/test/"

def resize_img(img, new_height):
    # Convert the image to PIL Image
    img = Image.fromarray(img)

    # get width of resized image
    new_width = int(img.width * new_height / img.height)

    # Resize the image using PIL
    img = img.resize((new_width, new_height)) 

    # Convert the resized image back to numpy array
    img = np.array(img)                            

    return img

def resize_gif(gif, new_height):
    imgs = []
    for img in gif:
        img = resize_img(img, new_height)
        imgs.append(img)
    return np.array(imgs)

def create_gif_array(folder_fit_path, img_name, gif_path, src_path):
    NEW_HEIGHT = 512

    # Get all gif files in the folder
    frames = imageio.v3.imread(gif_path + '.gif', index=None)
    frames = resize_gif(frames, NEW_HEIGHT)
    num_frames = frames.shape[0]

    img_overlay = imageio.v3.imread(gif_path + '_src.png', index=None)
    img_orig = imageio.v3.imread(src_path, index=None)

    frames = np.array(frames)
    # reshape img_orig and img_overlay to height of frame height (1024)

    # Convert the image to PIL Image
    img_overlay = resize_img(img_overlay, NEW_HEIGHT)  
    img_orig = resize_img(img_orig, NEW_HEIGHT)                          

    # repeat img of shape (width, height, channel) to be (num_gifs, height, width, num_frames, channels)
    img_overlay = np.repeat(img_overlay[None], num_frames, axis=0)
    img_orig = np.repeat(img_orig[None], num_frames, axis=0)

    # concatenate imgs und frames along width
    array = np.concatenate((img_orig, img_overlay, frames), axis=2)

    # Save the array gif
    output_folder = folder_fit_path + '_stacked'
    os.makedirs(output_folder, exist_ok=True)
    imageio.mimwrite(output_folder + f'/{img_name}.gif', array, fps=8)

def compare_fits(base_path, img_name='boys_2190_0', f1='fit_pseudogt', f2='fit_diffprior'):
    NEW_HEIGHT = 512

    # Get all gif files in the folder
    frames1 = imageio.v3.imread(f'{base_path}/{f1}/{img_name}/{img_name}.gif', index=None)
    frames2 = imageio.v3.imread(f'{base_path}/{f2}/{img_name}/{img_name}.gif', index=None)
    frames1 = resize_gif(frames1, NEW_HEIGHT)
    frames2 = resize_gif(frames2, NEW_HEIGHT)
    num_frames = frames1.shape[0]

    # get the original image
    src_img_name = '_'.join(img_name.split('_')[:-1])
    img_orig = imageio.v3.imread(f'{base_path}/src_images/{src_img_name}.png', index=None)
    img_orig = resize_img(img_orig, NEW_HEIGHT)                          

    # repeat img of shape (width, height, channel) to be (num_gifs, height, width, num_frames, channels)
    img_orig = np.repeat(img_orig[None], num_frames, axis=0)

    # concatenate imgs und frames along width
    array = np.concatenate((frames1, img_orig, frames2), axis=2)

    # Save the array gif in new folder names f1_f2 under base_path
    output_folder = f'{base_path}{f1}_{f2}'
    os.makedirs(output_folder, exist_ok=True)
    imageio.mimwrite( f'{output_folder}/{img_name}.gif', array, fps=8)

# Call the function to create the gif array and save it as an mp4 file
fits = ['fit_pseudogt', 'fit_diffprior']
for fit in fits:
    folder_fit_path = os.path.join(folder_path, fit)
    img_names = os.listdir(folder_fit_path)
    img_names.sort()
    for img_name in img_names:
        src_path = os.path.join(folder_path, 'src_images', '_'.join(img_name.split('_')[:-1]) + '.png')
        gif_path = os.path.join(folder_fit_path, img_name, img_name)
        if not os.path.exists(folder_fit_path + '_stacked' + f'/{img_name}.gif'):
            print(folder_fit_path + '_stacked' + f'/{img_name}.gif')
            create_gif_array(folder_fit_path, img_name, gif_path, src_path)

# comapre gifs 
fits1 = ['fit_diffprior']
fits2 = ['bev', 'fit_baseline']
for fit1 in fits1:
    for fit2 in fits2:
        if fit1 != fit2:
            folder_fit1_path = os.path.join(folder_path, fit1)
            img_names = os.listdir(folder_fit1_path)
            img_names.sort()
            for img_name in img_names:
                if not os.path.exists(f'{folder_path}/{fit1}_{fit2}/{img_name}.gif'):
                    print(f'{folder_path}{fit1}_{fit2}/{img_name}.gif')
                    compare_fits(folder_path, img_name, fit1, fit2)