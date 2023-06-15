import os
import imageio
import numpy as np
from PIL import Image, ImageSequence

# Specify the folder path, number of rows, and number of columns
folder_path = "media/projectpage/sample"
rows = 4
columns = 5

def create_gif_array(folder_path, rows, columns):
    # Get all gif files in the folder
    gif_files = [file for file in os.listdir(folder_path) if file.endswith(".gif")]
    num_gifs = rows * columns

    # Sort the files in alphabetical order
    gif_files.sort()

    # Create an empty list to store the frames
    frames = []

    for gif_idx, file in enumerate(gif_files[:num_gifs]):
        # Read the GIF file using imageio
        gif_path = os.path.join(folder_path, file)
        gif_frames = imageio.v3.imread(gif_path, index=None)

        # Append the frames to the list
        frames.append(gif_frames)

    frames = np.array(frames)
    # reorder dimsions in frames to be (num_gifs, height, width, num_frames, channels)
    frames = np.transpose(frames, (0,2,3,1,4)).reshape(rows, columns, 256, 200, 30, 3)

    # concatenate the first four dimensions to create array of shape 3 * 256, 5 * 200, 30, 3
    array = np.concatenate(np.concatenate(frames, axis=1), axis=1)
    array = np.transpose(array, (2,0,1,3))

    # Save the array gif
    imageio.mimwrite(f"{folder_path}/samples.gif", array, fps=8)

# Call the function to create the gif array and save it as an mp4 file
create_gif_array(folder_path, rows, columns)
