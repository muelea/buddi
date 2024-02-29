import os.path as osp 
import json 
import torch
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
from loguru import logger as guru
from pytorch3d.transforms import matrix_to_axis_angle
import torch
import torch.nn as nn

CHI3D_OPENPOSE_MAP = [9,8,14,15,16,11,12,13,0,4,5,6,1,2,3]

def project_3d_to_2d(points3d, intrinsics, intrinsics_type):
    if intrinsics_type == 'w_distortion':
        p = intrinsics['p'][:, [1, 0]]
        x = points3d[:, :2] / points3d[:, 2:3]
        r2 = np.sum(x**2, axis=1)
        radial = 1 + np.transpose(np.matmul(intrinsics['k'], np.array([r2, r2**2, r2**3])))
        tan = np.matmul(x, np.transpose(p))
        xx = x*(tan + radial) + r2[:, np.newaxis] * p
        proj = intrinsics['f'] * xx + intrinsics['c']
    elif intrinsics_type == 'wo_distortion':
        xx = points3d[:, :2] / points3d[:, 2:3]
        proj = intrinsics['f'] * xx + intrinsics['c']
    return proj

def plot_over_image(frame, points_2d=[], path_to_write=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(frame)
    cold = {0: 'white', 1:'red', 2:'blue', 3:'yellow', 4:'green'}
    for midx, method in enumerate(points_2d):
        ax.plot(method[:, 0], method[:, 1], 'x', markeredgewidth=10, color=cold[midx])
            
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if path_to_write:
        plt.ioff()
        plt.savefig(path_to_write, pad_inches = 0, bbox_inches='tight')

class CHI3DProjectJoints():
    
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
            'chi3d': {},
        }
    
    
    def load_single_image(self, subject, action, annotation):
        
        # annotation / image paths
        orig_subj_folder = osp.join(self.original_data_folder, self.split_folder, subject)
        cameras = os.listdir(osp.join(orig_subj_folder, 'camera_parameters'))
        processed_subj_folder = osp.join(self.processed_data_folder, self.split_folder, subject)
        frame_id = annotation['fr_id']

        joints_3d_path = osp.join(orig_subj_folder, self.joints3d_folder, f'{action}.json')
        joints_3d = np.array(json.load(open(joints_3d_path, 'r'))['joints3d_25'])[:, frame_id, :]

        for cam in cameras:
            imgname = f'{action}_{frame_id:06d}_{cam}'
            img_path = osp.join(orig_subj_folder, self.image_folder, f'{imgname}.jpg')

            # project joints to 2d
            cam_path = osp.join(orig_subj_folder, 'camera_parameters', cam, f'{action}.json')
            with open(cam_path) as f:
                cam_params = json.load(f)
                for key1 in cam_params:
                    for key2 in cam_params[key1]:
                        cam_params[key1][key2] = np.array(cam_params[key1][key2])  

            j3d = joints_3d #[frame_id]
            j3d_in_camera = np.matmul(np.array(j3d) - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R']))
            j2d_camera_h0 = project_3d_to_2d(j3d_in_camera[0], cam_params['intrinsics_w_distortion'], 'w_distortion')
            j2d_camera_h1 = project_3d_to_2d(j3d_in_camera[1], cam_params['intrinsics_w_distortion'], 'w_distortion')
            # concat joints 
            j2d_camera = np.concatenate([j2d_camera_h0[None], j2d_camera_h1[None]], axis=0)
            # append ones to j2d along axis -1
            j2d_camera = np.concatenate([j2d_camera, np.ones((2, 25, 1))], axis=-1)
            
            # render overlay for debugging
            # IMG = cv2.imread(img_path)
            # plot_over_image(IMG, points_2d=[j2d_camera[0], j2d_camera[1]], path_to_write=f'outdebug/chi3d_joints2d.png')
                
            # add to full clean dataset
            #self.output[subject][imgname] = {}

            # to openpose format
            op_j2d_camera = j2d_camera.copy()
            op_j2d_camera[:, :15, :] = j2d_camera[:, CHI3D_OPENPOSE_MAP, :]

            if not subject in self.output['openpose_coco25']:
                self.output['openpose_coco25'][subject] = {}
                self.output['chi3d'][subject] = {}

            self.output['openpose_coco25'][subject][imgname] = op_j2d_camera
            self.output['chi3d'][subject][imgname] = j2d_camera
        


    def process(self):

        processed_data_path = osp.join(
            self.processed_data_folder, self.split_folder, f'images_contact_projected_joints_2d.pkl'
        )

        guru.info(f'Processing data from {self.original_data_folder}')

        # iterate though dataset / images
        for pair in os.listdir(osp.join(self.original_data_folder, self.split_folder)):
            self.output[pair] = {}
            pair_folder = osp.join(self.original_data_folder, self.split_folder, pair)
            annotation_fn = osp.join(pair_folder, 'interaction_contact_signature.json')
            annotation = json.load(open(annotation_fn, 'r'))

            for imgname, anno in tqdm(annotation.items()):
                self.load_single_image(pair, imgname, anno)

        # save data to processed data folder
        with open(processed_data_path, 'wb') as f:
            pickle.dump(self.output, f)

     

if __name__ == "__main__":
   
    original_data_folder = 'datasets/original/CHI3D'
    processed_data_folder = 'datasets/processed/CHI3D'
    split = 'train'

    chi3d_data = CHI3DProjectJoints(
        original_data_folder,
        processed_data_folder,
        image_folder='images',
        split=split,
    )
   
    chi3d_data.process()
