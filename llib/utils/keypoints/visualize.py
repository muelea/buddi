import cv2
import numpy as np
import json
import pickle
import os
import matplotlib.cm as cm
from llib.utils.keypoints.loader import load_keypoints

def get_unique_color(person_index, total_people):
    """
    Get a unique color for each person based on their index.

    Parameters:
        person_index (int): The index of the person in the list of keypoints.
        total_people (int): The total number of people in the list.

    Returns:
        tuple: A (R, G, B) color tuple.
    """
    color_map = cm.get_cmap('tab20')  # You can change the colormap as needed
    color_index = person_index % color_map.N
    color = tuple(int(255 * x) for x in color_map(color_index)[:3])
    return color

def read_pose_tree_connections(file_path):
    """
    Read the tree connection from a text file.

    Parameters:
        file_path (str): The path to the text file containing the tree connections.

    Returns:
        list of tuples: List of (index1, index2) pairs representing the tree connections.
    """
    tree_connections = []
    with open(file_path, 'r') as file:
        for line in file:
            index1, index2 = map(int, line.strip().split())
            tree_connections.append((index1, index2))
    return tree_connections


def visualize_keypoints_with_tree(image, keypoints, tree_connections, confidence=None, color=(0, 0, 255)):
    """
    Visualizes an image with 2D keypoints and a tree connecting the keypoints.

    Parameters:
        image (numpy.ndarray): The input image.
        keypoints (list of tuples): List of (x, y) coordinates representing the keypoints.
        tree_connections (list of tuples): List of (index1, index2) pairs, connecting keypoints with an edge.

    Returns:
        numpy.ndarray: The image with keypoints and tree connections visualized.
    """
    # Create a copy of the image to draw keypoints and the tree on
    image_with_keypoints = image.copy()

    # Draw keypoints on the image
    for idx, (x, y) in enumerate(keypoints):
        if (x != 0 and y != 0):
            if confidence is None:
                cv2.circle(image_with_keypoints, (int(x), int(y)), 2, color, -1)
            else:
                if confidence[idx] > 0.5:
                    cv2.circle(image_with_keypoints, (int(x), int(y)), 2, color, -1)

    # Draw tree connections on the image
    for index1, index2 in tree_connections:
        x1, y1 = keypoints[index1]
        x2, y2 = keypoints[index2]
        if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
            if confidence is None:
                cv2.line(image_with_keypoints, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            else:
                if confidence[index1] > 0.3 and confidence[index2] > 0.3:
                    cv2.line(image_with_keypoints, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    return image_with_keypoints

def visualize_all_people_keypoints(image, all_keypoints, tree_connections, confidence=None):
    """
    Visualizes keypoints of all people in an image.

    Parameters:
        image (numpy.ndarray): The input image.
        all_keypoints (numpy.ndarray): P x K x 2 array containing keypoints of all people.
                                      P is the number of people, K is the number of keypoints,
                                      and each keypoint is represented as a (x, y) coordinate.
        tree_connections (list of tuples): List of (index1, index2) pairs, connecting keypoints with an edge.

    Returns:
        numpy.ndarray: The image with keypoints of all people and tree connections visualized.
    """
    image_with_keypoints = image.copy()

    for person_idx, person_keypoints in enumerate(all_keypoints):
        color = get_unique_color(person_idx, all_keypoints.shape[0])
        person_confidence = None if confidence is None else confidence[person_idx]
        result_image = visualize_keypoints_with_tree(image_with_keypoints, person_keypoints, tree_connections, person_confidence, color)
        image_with_keypoints = result_image

    return image_with_keypoints

def visulize_keypoints(image_name, image_input_folder, keypoint_folder, keypoint_detector, output_folder):
    image_path = f'{image_input_folder}/{image_name}.png'
    image = cv2.imread(image_path)

    # read keypoints
    ext = 'pkl' if keypoint_detector == 'vitposeplus' else 'json'
    keypoints_path = f'{keypoint_folder}/{image_name}.{ext}'
    input_format = 'coco-133' if 'vitposeplus' in keypoints_path else 'coco-25'
    keypoints = load_keypoints(keypoints_path, input_format, 'coco-25', True, True)

    # read tree connections
    if input_format == 'coco-133':
        file_path = 'essentials/body_model_utils/vitposeplus_tree_connection.txt'
    else:
        file_path = 'essentials/body_model_utils/openpose_tree_connection.txt'
    tree_connections = read_pose_tree_connections(file_path)

    # Visualize the image with keypoints and tree connections
    result_image = visualize_all_people_keypoints(image, keypoints[:,:,:2], tree_connections, keypoints[:,:,2])
    cv2.imwrite(f'{output_folder}/{image_name}_{keypoint_detector}.png', result_image)

def visualize_dataset(image_input_folder, keypoint_folder, keypoint_detector, output_folder):
    image_names = os.listdir(image_input_folder)
    for image_name in sorted(image_names)[:100]:
        image_name = image_name.split('.')[0]
        print(image_name)
        visulize_keypoints(image_name, image_input_folder, keypoint_folder, keypoint_detector, output_folder)

if __name__ == "__main__":
    
    # Visaulize example image and keypoints
    image_name = 'college_232334'
    image_input_folder = 'demo/data/FlickrCI3D_Signatures/demo/images'
    output_folder = 'outdebug'

    visulize_keypoints(
        image_name, 
        image_input_folder=image_input_folder, 
        keypoint_folder='demo/data/FlickrCI3D_Signatures/demo/keypoints/vitposeplus', 
        keypoint_detector='vitposeplus', 
        output_folder=output_folder
    )

    visulize_keypoints(
        image_name, 
        image_input_folder=image_input_folder, 
        keypoint_folder='demo/data/FlickrCI3D_Signatures/demo/keypoints/keypoints', 
        keypoint_detector='openpose', 
        output_folder=output_folder
    )

    # visualize FlickrCI3D dataset
    os.makedirs('datasets/processed/FlickrCI3D_Signatures/train/keypoints/vitposeplus_openpose_vis', exist_ok=True)
    visualize_dataset(
        image_input_folder='datasets/original/FlickrCI3D_Signatures/train/images',
        keypoint_folder='datasets/processed/FlickrCI3D_Signatures/train/keypoints/vitposeplus',
        keypoint_detector='vitposeplus',
        output_folder='datasets/processed/FlickrCI3D_Signatures/train/keypoints/vitposeplus_openpose_vis'
    )

    visualize_dataset(
        image_input_folder='datasets/original/FlickrCI3D_Signatures/train/images',
        keypoint_folder='datasets/processed/FlickrCI3D_Signatures/train/keypoints/openpose',
        keypoint_detector='openpose',
        output_folder='datasets/processed/FlickrCI3D_Signatures/train/keypoints/vitposeplus_openpose_vis'
    )