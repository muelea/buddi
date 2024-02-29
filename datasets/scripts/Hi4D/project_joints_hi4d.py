import os.path as osp 
import json 
import torch
import smplx
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
from loguru import logger as guru
from pytorch3d.transforms import matrix_to_axis_angle
import torch
import torch.nn as nn

SMPL_IDXS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
HI4D_OPENPOSE_MAP = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
HI4D_OPENPOSE_MAP = HI4D_OPENPOSE_MAP[1:15]


def plot_over_image(frame, points_2d=[], path_to_write=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(frame)
    cold = {0: 'blue', 1:'red', 2:'pink', 3:'yellow', 4:'green'}
    for midx, method in enumerate(points_2d):
        ax.plot(method[:, 0], method[:, 1], 'x', markeredgewidth=10, color=cold[midx])
        for idx, point in enumerate(points_2d[0]):
            ax.text(point[0], point[1], str(idx))
            
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if path_to_write:
        plt.ioff()
        plt.savefig(path_to_write, pad_inches = 0, bbox_inches='tight')


class Hi4DProjectJoints():
    
    BEV_FOV = 60

    def __init__(
        self,
        original_data_folder,
        processed_data_folder,
        image_folder,
        joints3d_folder='joints3d_25',
        split='train',
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder
        self.split = split
        self.split_folder = 'test' if split == 'test' else 'train'  

        self.joints3d_folder = joints3d_folder
        self.image_folder = image_folder

        # output format 
        self.output = {
            'openpose_coco25': {},
            'hi4d': {},
        }
    
    
    def load_single_image(self, imgname, img_path, pair, action, cam, smpl_data, camera_data): #subject, action, annotation):
        
        imgname_raw = imgname.split('.')[0]
        j3d = smpl_data['joints_3d']

        RR = camera_data['extrinsics'][:,:,:3]
        TT = camera_data['extrinsics'][:,:,3]
        KK = camera_data['intrinsics']

        j3d_camera = np.transpose(np.matmul(RR, np.transpose(j3d, (0,2,1))), (0,2,1)) + TT
        j2d_camera = np.transpose(
            np.matmul(KK[0], np.transpose(j3d_camera, (0,2,1))),
        (0,2,1))
        j2d_camera = j2d_camera[:, :, :2] / j2d_camera[:, :, 2:3]
        
        # append ones to j2d along axis -1
        j2d_camera = np.concatenate([j2d_camera, np.ones((2, 24, 1))], axis=-1)
        # render overlay for debugging
        #IMG = cv2.imread(img_path)
        #plot_over_image(IMG, points_2d=[j2d_camera[0], j2d_camera[1]], 
        #    path_to_write=f'outdebug/hi4d_joints2d.png')

        # to openpose format
        op_j2d_camera = np.zeros((2,25,3), dtype=j2d_camera.dtype)
        op_j2d_camera[:, SMPL_IDXS, :] = j2d_camera[:, HI4D_OPENPOSE_MAP, :]
        op_j2d_camera[:,:,-1] = 0.0
        op_j2d_camera[:, SMPL_IDXS, -1] = 1.0

        # render overlay for debugging
        #IMG = cv2.imread(img_path)
        #plot_over_image(IMG, points_2d=[op_j2d_camera[0], op_j2d_camera[1]], 
        #    path_to_write=f'outdebug/hi4d_joints2d_openpose.png')

        if not pair in self.output['openpose_coco25']:
            self.output['openpose_coco25'][pair] = {}
            self.output['hi4d'][pair] = {}
        
        if not action in self.output['openpose_coco25'][pair]:
            self.output['openpose_coco25'][pair][action] = {}
            self.output['hi4d'][pair][action] = {}

        if not cam in self.output['openpose_coco25'][pair][action]:
            self.output['openpose_coco25'][pair][action][cam] = {}
            self.output['hi4d'][pair][action][cam] = {}

        self.output['openpose_coco25'][pair][action][cam][imgname_raw] = op_j2d_camera
        self.output['hi4d'][pair][action][cam][imgname_raw] = j2d_camera
        


    def process(self):

        processed_data_path = osp.join(
            self.processed_data_folder, f'images_contact_projected_joints_2d.pkl'
        )

        guru.info(f'Processing data from {self.original_data_folder}')


        # iterate though dataset / images
        for pair in os.listdir(osp.join(self.original_data_folder)):
            if not 'pair' in pair:
                continue
            print(pair)

            pair_folder = osp.join(self.original_data_folder, pair)
            for action in os.listdir(pair_folder):
                if action.startswith('.'):
                    continue

                action_folder = osp.join(pair_folder, action)
                camera_data = np.load(osp.join(action_folder, 'cameras/rgb_cameras.npz'))
    
                # read smpl data 
                smpl_data = {}
                for frame in os.listdir(osp.join(action_folder, 'smpl')):
                    smpl_data[frame.split('.')[0]] = np.load(osp.join(action_folder, 'smpl', frame))

                for cam in os.listdir(osp.join(action_folder, 'images')):
                    if cam.startswith('.'):
                        continue
                    camera_image_folder = osp.join(action_folder, 'images', cam)
                    def get_current_cam(camera_data, cam_idx):
                        array_idx = np.where(camera_data['ids'] == int(cam_idx))
                        out = {}
                        for key in camera_data.keys():
                            out[key] = camera_data[key][array_idx]
                        return out

                    curr_cam_data = get_current_cam(camera_data, cam)
                    for imgname in os.listdir(camera_image_folder):
                        if not imgname.endswith('jpg'):
                            continue
                        img_path = osp.join(camera_image_folder, imgname)
                        curr_smpl_data = smpl_data[imgname.split('.')[0]]
                        self.load_single_image(imgname, img_path, pair, action, cam, curr_smpl_data, curr_cam_data)

        # save data to processed data folder
        with open(processed_data_path, 'wb') as f:
            pickle.dump(self.output, f)

     

if __name__ == "__main__":
   
    original_data_folder = 'datasets/original/Hi4D'
    processed_data_folder = 'datasets/processed/Hi4D'
    split = 'train'

    chi3d_data = Hi4DProjectJoints(
        original_data_folder,
        processed_data_folder,
        image_folder='images',
    )
   
    chi3d_data.process()
