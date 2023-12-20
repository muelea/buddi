import json
import pickle
import numpy as np

# OpenPose Full body has 135 joints 
# 25 body, 21 left hand, 21 right hand, 70 face (137 total)

# Coco whole body has 133 joints
# 23 body, 21 left hand, 21 right hand, 68 face (133 total)
# body misses throat and mid hip joints
# face does not have left/right pupile landmarks

def load_keypoints(file_path, input_format='coco-25', output_format='coco-25', include_hands=False, include_face=False):
    """
    Load keypoints from either an OpenPose JSON file or a pickle file.

    Parameters:
        file_path (str): The path to the input file.
        input_format (str): The input format for keypoints. Options: 'coco-25' (default), 'coco-133'
        output_format (str): The output format for keypoints. Options: 'coco-25' (default)
        include_hands (bool): Whether to include hand keypoints (if available).
        include_face (bool): Whether to include face keypoints (if available).

    Returns:
        numpy.ndarray: P x K x 2 array containing keypoints of all people.
                       P is the number of people, K is the number of keypoints,
                       and each keypoint is represented as a (x, y) coordinate.
    """
    # Get the file extension
    file_extension = file_path.split('.')[-1].lower()

    # Load keypoints based on the file extension
    if file_extension == 'json':
        return load_keypoints_from_json(file_path, input_format, output_format, include_hands, include_face)
    elif file_extension == 'pkl':
        return load_keypoints_from_pickle(file_path, input_format, output_format, include_hands, include_face)
    else:
        raise ValueError("Unsupported file format. Only JSON and pickle files are supported.")

def load_keypoints_from_json(file_path, input_format='coco-25', output_format='coco-25', include_hands=False, include_face=False):
    """
    Load keypoints from an OpenPose JSON file.

    Parameters:
        file_path (str): The path to the OpenPose JSON file.
        input_format (str): The input format for keypoints. Options: 'coco-25' (default), 'coco-133'
        output_format (str): The output format for keypoints. Options: 'coco-25' (default)
        include_hands (bool): Whether to include hand keypoints (if available).
        include_face (bool): Whether to include face keypoints (if available).

    Returns:
        numpy.ndarray: P x K x 2 array containing keypoints of all people.
                       P is the number of people, K is the number of keypoints,
                       and each keypoint is represented as a (x, y) coordinate.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    if output_format != 'coco-25':
        raise ValueError("Invalid output_format. Options: 'coco-25'")
    
    if input_format not in ['coco-25']:
        raise ValueError("Invalid input_format. Options: 'coco-25'")
    keypoints = []

    for person in data['people']:
        pose_keypoints = person['pose_keypoints_2d']
        if include_hands:
            hand_left_keypoints = person.get('hand_left_keypoints_2d', [])
            hand_right_keypoints = person.get('hand_right_keypoints_2d', [])
            pose_keypoints.extend(hand_left_keypoints)
            pose_keypoints.extend(hand_right_keypoints)
        if include_face:
            face_keypoints = person.get('face_keypoints_2d', [])
            pose_keypoints.extend(face_keypoints)

        keypoints.append(np.array(pose_keypoints).reshape(-1, 3))

    return np.array(keypoints)

def load_keypoints_from_pickle(file_path, input_format='coco-25', output_format='coco-25', include_hands=False, include_face=False):
    """
    Load keypoints from a pickle file, e.g. written from ViTPosePlus.

    Parameters:
        file_path (str): The path to the pickle file.
        input_format (str): The input format for keypoints. Options: 'coco-25' (default), 'coco-133'
        output_format (str): The output format for keypoints. Options: 'coco-25' (default)
        include_hands (bool): Whether to include hand keypoints (if available).
        include_face (bool): Whether to include face keypoints (if available).

    Returns:
        numpy.ndarray: P x K x 2 array containing keypoints of all people.
                       P is the number of people, K is the number of keypoints,
                       and each keypoint is represented as a (x, y) coordinate.
    """
    with open(file_path, 'rb') as file:
        keypoints = pickle.load(file)

    if output_format != 'coco-25':
        raise ValueError("Invalid output_format. Options: 'coco-25'")
    
    if input_format not in ['coco-133']:
        raise ValueError("Invalid input_format. Options: 'coco-133'")

    # keypoints to np array
    keypoints = np.array([x['keypoints'] for x in keypoints])

    # output format
    num_people = keypoints.shape[0]
    num_keypoints_out = 25
    if include_hands:
        num_keypoints_out += 2*21
    if include_face:
        num_keypoints_out += 70
    keypoints_out = np.zeros((num_people, num_keypoints_out, 3))
    
    # write keypoints to output
    keypoints_out[:,[0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]] = keypoints[:,:17]
    keypoints_out[:,19:25] = keypoints[:,17:23] # left and right foot

    if include_hands: 
        keypoints_out[:,25:46] = keypoints[:,-42:-21] # left hand
        keypoints_out[:,46:67] = keypoints[:,-21:] # right hand
    
    if include_face:
        keypoints_out[:,67:135] = keypoints[:,23:-42] # face

    return keypoints_out