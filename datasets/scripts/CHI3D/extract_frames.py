import cv2
import os
import json
import shutil
import os.path as osp
import argparse 

def read_video(vid_path):
    frames = []
    cap = cv2.VideoCapture(vid_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def copy_train_contact_frames(input_data_folder, output_data_folder, sequence):
    """ extract training images"""
    annotation_fn = osp.join(input_data_folder, f'train/{sequence}/interaction_contact_signature.json')
    annotation = json.load(open(annotation_fn, 'r'))
    camera_folder = osp.join(input_data_folder, f'train/{sequence}/videos')
    for cam in sorted(os.listdir(camera_folder)):
        actions_folder = f'{camera_folder}/{cam}'
        input_folder = osp.join(output_data_folder, f'train/{sequence}/images')
        output_folder = osp.join(output_data_folder, f'train/{sequence}/images_contact')
        os.makedirs(output_folder, exist_ok=True)

        for action_fn in sorted(os.listdir(f'{camera_folder}/{cam}')):
            if action_fn.startswith('.'):
                continue
        
            action, ending = action_fn.split('.')
            contact_frame = annotation[action]['fr_id']
            existing_path = f'{input_folder}/{action}_{contact_frame:06d}_{cam}.jpg'
            new_path = f'{output_folder}/{action}_{contact_frame:06d}_{cam}.jpg'
            if osp.exists(existing_path):
                shutil.copy(existing_path, new_path)

def copy_test_contact_frames(input_data_folder, output_data_folder, sequence):
    """ extract test images"""
    annotation_fn = osp.join(input_data_folder, f'test/{sequence}/interaction_contact_signature.json')
    annotation = json.load(open(annotation_fn, 'r'))
    input_folder = osp.join(output_data_folder, f'test/{sequence}/images')
    output_folder = osp.join(output_data_folder, f'test/{sequence}/images_contact')
    os.makedirs(output_folder, exist_ok=True)
    for action_fn in sorted(os.listdir(actions_folder)):
        action, ending = action_fn.split('.')
        contact_frame = annotation[action]['fr_id']
        existing_path = f'{input_folder}/{action}_{idx:06d}.jpg'
        new_path = f'{output_folder}/{action}_{idx:06d}.jpg'
        if osp.exists(existing_path):
            shutil.copy(existing_path, new_path)

def extract_train_frames(input_data_folder, output_data_folder, sequence):
    """ extract training images"""
    camera_folder = osp.join(input_data_folder, f'train/{sequence}/videos')
    for cam in sorted(os.listdir(camera_folder)):
        actions_folder = f'{camera_folder}/{cam}'
        output_folder = osp.join(output_data_folder, f'train/{sequence}/images')
        os.makedirs(output_folder, exist_ok=True)
        for action_fn in sorted(os.listdir(f'{camera_folder}/{cam}')):
            if action_fn.startswith('.'):
                continue
            action, ending = action_fn.split('.')

            frames = read_video(f'{camera_folder}/{cam}/{action_fn}')

            check_idx = int(len(frames) - 1)
            if osp.exists(f'{output_folder}/{action}_{check_idx:06d}_{cam}.jpg'):
                continue

            print(f'Writing: {camera_folder}/{cam}/{action}')
            for idx, frame in enumerate(frames):
                cv2.imwrite(f'{output_folder}/{action}_{idx:06d}_{cam}.jpg', frame)

def extract_test_frames(input_data_folder, output_data_folder, sequence):
    """ extract test images"""
    actions_folder = osp.join(input_data_folder, f'test/{sequence}/videos')
    output_folder = osp.join(output_data_folder, f'test/{sequence}/images')
    os.makedirs(output_folder, exist_ok=True)
    for action_fn in sorted(os.listdir(actions_folder)):
        action, ending = action_fn.split('.')

        frames = read_video(f'{actions_folder}/{action_fn}')

        check_idx = int(len(frames) - 1)
        if osp.exists(f'{output_folder}/{action}_{check_idx:06d}.jpg'):
            continue

        print(f'Writing: {actions_folder}/{action}')
        for idx, frame in enumerate(frames):
            cv2.imwrite(f'{output_folder}/{action}_{idx:06d}.jpg', frame)

def wrapper(func, input_data_folder, output_data_folder, sequence):
    func(input_data_folder, output_data_folder, sequence)

if __name__ == '__main__':
    """ Read videos from input_folder and extract frames to output_folderoutput_folder """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='datasets/original/CHI3D',
        help='Root folder where the original CHI3D data is stored')
    parser.add_argument('--output_folder', type=str, default='datasets/processed/CHI3D',
        help='Root folder where the images will be saved')
    parser.add_argument('--sequence', type=str, default='all',
        help='Sequence to extract')
    parser.add_argument('--use_multiprocessing', action='store_true', 
        help='Use multiprocessing to extract frames')
    parser.add_argument('--contact-only', action='store_true', 
        help='Only contact frames.')
    args = parser.parse_args()

    sequence = args.sequence
    use_multiprocessing = args.use_multiprocessing
    contact_only = args.contact_only


    TRAIN_SEQUENCES = ['s02', 's03', 's04']
    TEST_SEQUENCES = ['s01', 's05']

    if use_multiprocessing and sequence == 'all':
        from multiprocess import Pool
        p = Pool(5)
        if contact_only:
            train_args = [(copy_train_contact_frames, args.input_folder, args.output_folder, seq) for seq in TRAIN_SEQUENCES]
            test_args = [(copy_test_contact_frames, args.input_folder, args.output_folder, seq) for seq in TEST_SEQUENCES]
        else:
            train_args = [(extract_train_frames, args.input_folder, args.output_folder, seq) for seq in TRAIN_SEQUENCES]
            test_args = [(extract_test_frames, args.input_folder, args.output_folder, seq) for seq in TEST_SEQUENCES]
        p.starmap(wrapper, train_args + test_args)

    if sequence == 'all':
        # extract train frames
        for seq in TRAIN_SEQUENCES:
            if contact_only:
                copy_train_contact_frames(args.input_folder, args.output_folder, seq)
            else:
                extract_train_frames(args.input_folder, args.output_folder, seq)
        # extract test frames
        for seq in TEST_SEQUENCES:
            if contact_only:
                copy_test_contact_frames(args.input_folder, args.output_folder, seq)
            else:
                extract_test_frames(args.input_folder, args.output_folder, seq)
    else:
        if sequence in TRAIN_SEQUENCES:
            if contact_only:
                copy_train_contact_frames(args.input_folder, args.output_folder, sequence)
            else:
                extract_train_frames(args.input_folder, args.output_folder, sequence)
        elif sequence in TEST_SEQUENCES:
            if contact_only:
                copy_test_contact_frames(args.input_folder, args.output_folder, sequence)
            else:
                extract_test_frames(args.input_folder, args.output_folder, sequence)
        else:
            raise ValueError('Invalid sequence')