
from __future__ import division

import os
import torch
import os.path as osp
from torch.utils.data import Dataset
import numpy as np

class SingleOptiDataset(Dataset):

    def __init__(
             self,
             dataset_cfg,
             dataset_name,
             image_processing,
             split='train',
             body_model_type='smplx',
             use_hands=False,
             use_face=False,
             use_face_contour=False,
        ):
        """
        Base Dataset Class for optimization.
        Parameters
        ----------
        dataset_cfg: cfg
            config file of dataset
        dataset_name: str
            name of dataset (e.g. flickrci3ds)
        image_processing: cfg
            config file of image processing
        split: str
            split of dataset (train, val, test)
        body_model_type: str
            type of body model
        """

        super(SingleOptiDataset, self).__init__()

        self.image_processing = image_processing
        self.body_model_type = body_model_type
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.init_method = 'bev' 

        self.num_pose_params = 72
        self.num_shape_params = 10
        self.num_global_orient_params = 3
        self.num_transl_params = 3
        self.num_gt_kpts = 24
        self.num_op_kpts = 25

        self.kpts_idxs = np.arange(0,25)
        self.use_hands = use_hands
        self.use_face = use_face
        self.use_face_contour = use_face_contour

        if use_hands:
            self.kpts_idxs = np.concatenate([self.kpts_idxs, np.arange(25, 25 + 2 * 21)])
        if use_face:
            self.kpts_idxs = np.concatenate([self.kpts_idxs, np.arange(67, 67 + 51)])
        if use_face_contour:
            self.kpts_idxs = np.concatenate([self.kpts_idxs, np.arange(67 + 51, 67 + 51 + 17)])

        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.body_model_type = body_model_type

        self.img_dir = osp.join(
            dataset_cfg.original_data_folder, dataset_cfg.image_folder
        )

        self.data = self.load_data()
        self.len = len(self.data)


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_data(self):
        if self.dataset_name in ["flickrci3ds", "flickrci3ds_adult", "flickrci3ds_child"]:
            from .preprocess.flickrci3d_signatures_contacts import FlickrCI3D_Signatures
            dataset = FlickrCI3D_Signatures(
                **self.dataset_cfg, 
                split=self.split,
                body_model_type=self.body_model_type
            ).load(
                load_from_scratch=False,
                allow_missing_information=True,
                processed_fn_ext='_optimization.pkl'            
            )
        elif self.dataset_name == 'chi3d':
            from .preprocess.chi3d import CHI3D
            dataset = CHI3D(
                **self.dataset_cfg, 
                split=self.split,
                body_model_type=self.body_model_type
            ).load(
                load_from_scratch=False, 
                allow_missing_information=True,
                processed_fn_ext='_optimization.pkl',
            )
        elif self.dataset_name == 'demo':
            from .preprocess.demo import Demo
            dataset = Demo(
                **self.dataset_cfg, 
                split=self.split,
                body_model_type=self.body_model_type
            ).load()
        elif self.dataset_name == 'hi4d':
            from .preprocess.hi4d import HI4D
            dataset = HI4D(
                **self.dataset_cfg, 
                split=self.split,
                body_model_type=self.body_model_type
            ).load(
                load_from_scratch=False, 
                allow_missing_information=True,
                processed_fn_ext='_optimization.pkl',
            )
        else:
            raise NotImplementedError
            
        return dataset


    def join_bbox(self, bb1, bb2):
        x1, y1 = min(bb1[0], bb2[0]), min(bb1[1], bb2[1])
        x2, y2 = max(bb1[2], bb2[2]), max(bb1[3], bb2[3])
        return [x1, y1, x2, y2]

        
    def to_tensors(self, target):
        for k, v in target.items():
            if isinstance(v, np.ndarray):
                target[k] = torch.from_numpy(v).float()
            elif isinstance(v, list):
                target[k] = torch.tensor(v).float()
            elif isinstance(v, dict):
                target[k] = self.to_tensors(v).float()
            elif isinstance(v, torch.Tensor):
                target[k] = v.float()
        return target


    def get_single_item(self, index):

        item = self.data[index]

        # crop image using both bounding boxes
        #h1_bbox, h2_bbox = item[f'bbox_h0'], item[f'bbox_h1']
        if 'flickr_bbox' in item.keys():
            bbox = item['flickr_bbox'].astype(int) 
        else:
            bbox = np.array([[0,0,0,0], [0,0,0,0]])
        
        bbox_join = self.join_bbox(bbox[0], bbox[1]) 
        # cast bbox to int
        bbox_join = np.array(bbox_join).astype(int)

        input_image = [0.0]
        img_height = item['img_height']
        img_width = item['img_width']

        if 'contact_index' in item.keys():
            contact_index = item['contact_index'] if item['contact_index'] is not None else 0
        else:
            contact_index = 0

        if 'img_out_fn' in item.keys():
            img_out_fn = item['img_out_fn']
        else:
            img_out_fn = item['imgname'].replace('.png', '_') + str(contact_index)
        
        gen_target = {
            'images': input_image,
            'imgpath': item['imgpath'],
            'contact_index': contact_index,
            'bbox': bbox,
            'bbox_h0h1': bbox_join,
            'imgname_fn': item['imgname'],
            'imgname_fn_out': img_out_fn,
            'img_height': img_height,
            'img_width': img_width,
            'sample_index': index,
        }

        cam_target = {
            'pitch': np.array(item['cam_rot'][0]),
            'yaw': np.array(item['cam_rot'][1]),
            'roll': np.array(item['cam_rot'][2]),
            'tx': np.array(item['cam_transl'][0]),
            'ty': np.array(item['cam_transl'][1]),
            'tz': np.array(item['cam_transl'][2]),
            'fl': np.array(item['fl']),
            'ih': np.array(item['img_height']),
            'iw': np.array(item['img_width']),
        }

        if 'contact_map' in item.keys():
            contact_map = item['contact_map']
        else:
            contact_map = np.zeros((75, 75)).astype(bool)

        if 'bev_smpl_vertices_root_trans' in item.keys():
            bev_smpl_vertices = item['bev_smpl_vertices_root_trans']
        else:
            bev_smpl_vertices = np.zeros((2, 6890, 3))

        op_keypoints = item[f'openpose'] 
        if 'vitpose' not in item.keys():
            vitpose_keypoints = np.zeros((2, 25, 3))
        else:
            vitpose_keypoints = item[f'vitpose']

        #### SELECT FINAL KEYPLOINTS ####   
        if self.use_hands:
            final_keypoints = item[f'vitposeplus']
            mask = item['vitposeplus_human_idx'] == -1 # use openpose for missing humans
            final_keypoints[mask] = item['openpose'][mask]
        else:
            final_keypoints = item[f'vitpose']
            mask = item['vitpose_human_idx'] == -1 # use openpose for missing humans
            final_keypoints[mask] = item['openpose'][mask]
            if np.unique(item['vitpose_human_idx']).shape[0] == 1: # and use openpose when vitpose would pick the same person twice
                final_keypoints = item[f'openpose']
            # add toe keypoints
            ankle_thres = 5.0
            right_ankle_residual = np.sum((final_keypoints[:,11,:] - op_keypoints[:,11,:])**2, axis=1)
            ram = right_ankle_residual < ankle_thres
            final_keypoints[ram,22,:] = op_keypoints[ram,22,:]
            left_ankle_residual = np.sum((final_keypoints[:,14,:] - op_keypoints[:,14,:])**2, axis=1)
            lam = left_ankle_residual < ankle_thres
            final_keypoints[lam,19,:] = op_keypoints[lam,19,:] 
        human_target = { 
            'contact_map': contact_map,
            'global_orient': item[f'{self.init_method}_{self.body_model_type}_global_orient'],
            'body_pose': item[f'{self.init_method}_{self.body_model_type}_body_pose'],
            'transl': item[f'{self.init_method}_{self.body_model_type}_transl'],
            'betas': item[f'{self.init_method}_{self.body_model_type}_betas'],
            'scale': item[f'{self.init_method}_{self.body_model_type}_scale'].astype(float),
            'vertices': item[f'{self.init_method}_{self.body_model_type}_vertices'],
            'bev_smpl_vertices': bev_smpl_vertices,
            'op_keypoints': op_keypoints[:,self.kpts_idxs,:],
            'vitpose_keypoints': vitpose_keypoints[:,self.kpts_idxs,:], 
            #'vitpose_keypoints': item[f'vitpose'][:,self.kpts_idxs,:],
            'vitposeplus_keypoints': item[f'vitposeplus'][:,self.kpts_idxs,:],
            'keypoints': final_keypoints[:,self.kpts_idxs,:], 
        }
                
        target = {**gen_target, **cam_target, **human_target}

        target = self.to_tensors(target)
        
        return target
