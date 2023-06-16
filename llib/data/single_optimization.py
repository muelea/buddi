
from __future__ import division

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

        self.num_pose_params = 72
        self.num_shape_params = 10
        self.num_global_orient_params = 3
        self.num_transl_params = 3
        self.num_gt_kpts = 24
        self.num_op_kpts = 25

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
            ).load()
        elif self.dataset_name == 'chi3d':
            from .preprocess.chi3d import CHI3D
            dataset = CHI3D(
                **self.dataset_cfg, 
                split=self.split,
                body_model_type=self.body_model_type
            ).load()
        elif self.dataset_name == 'demo':
            from .preprocess.demo import Demo
            dataset = Demo(
                **self.dataset_cfg, 
                split=self.split,
                body_model_type=self.body_model_type
            ).load()
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
                target[k] = torch.from_numpy(v)
            elif isinstance(v, list):
                target[k] = torch.tensor(v)
            elif isinstance(v, dict):
                target[k] = self.to_tensors(v)
        return target


    def get_single_item(self, index):

        item = self.data[index]

        # crop image using both bounding boxes
        h1_bbox, h2_bbox = item[f'bbox_h0'], item[f'bbox_h1']
        bbox = self.join_bbox(h1_bbox, h2_bbox) if h1_bbox is not None else None
        # cast bbox to int
        bbox = np.array(bbox).astype(int)
        h1_bbox = np.array(h1_bbox).astype(int)
        h2_bbox = np.array(h2_bbox).astype(int)
        
        input_image = [0.0]
        img_height = item['img_height']
        img_width = item['img_width']

        if 'contact_index' in item.keys():
            contact_index = item['contact_index'] if item['contact_index'] is not None else 0
        else:
            contact_index = 0

        img_out_fn = item['imgname'].replace('.png', '_') + str(contact_index)
        gen_target = {
            'images': input_image,
            'imgpath': item['imgpath'],
            'contact_index': contact_index,
            'bbox_h0': h1_bbox,
            'bbox_h1': h2_bbox,
            'bbox_h0h1': bbox,
            'imgname_fn': item['imgname'],
            'imgname_fn_out': img_out_fn,
            'img_height': img_height,
            'img_width': img_width,
            'sample_index': index,
        }

        if 'binary_contact' in item.keys():
            gen_target['binary_contact'] = item['binary_contact']

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

        if 'vertices_h0h1_contact_heat' in item.keys():
            bev_contact_heat = item['vertices_h0h1_contact_heat']
        else:
            bev_contact_heat = np.zeros((75, 75)).astype(np.float32)

        if 'contact_map' in item.keys():
            contact_map = item['contact_map']
        else:
            contact_map = np.zeros((75, 75)).astype(np.bool)
    

        h0id = 0 if item['transl_h0'][0] <= item['transl_h1'][0] else 1
        h1id = 1-h0id
        if h0id == 1:
            bev_contact_heat = bev_contact_heat.T
            contact_map = contact_map.T
            gen_target['bbox_h0'], gen_target['bbox_h1'] = gen_target['bbox_h1'], gen_target['bbox_h0']
        human_target = { 
            'contact_map': contact_map,
            'bev_contact_heat': bev_contact_heat, 
            'global_orient_h0': item[f'global_orient_h{h0id}'],
            'body_pose_h0': item[f'body_pose_h{h0id}'],
            'transl_h0': item[f'transl_h{h0id}'],
            'translx_h0': item[f'transl_h{h0id}'][[0]],
            'transly_h0': item[f'transl_h{h0id}'][[1]],
            'translz_h0': item[f'transl_h{h0id}'][[2]],
            'betas_h0': item[f'betas_h{h0id}'][:10],
            'scale_h0': item[f'betas_h{h0id}'][[-1]],
            'joints_h0': item[f'joints_h{h0id}'],
            'vertices_h0': item[f'vertices_h{h0id}'],
            'bev_keypoints_h0': item[f'bev_keypoints_h{h0id}'],
            'bev_orig_vertices_h0': item[f'bev_orig_vertices_h{h0id}'],
            'op_keypoints_h0': item[f'op_keypoints_h{h0id}'],
            'keypoints_h0': item[f'vitpose_keypoints_h{h0id}'], # added it's the used keypoints
            'vitpose_keypoints_h0': item[f'vitpose_keypoints_h{h0id}'],
            'global_orient_h1': item[f'global_orient_h{h1id}'],
            'body_pose_h1': item[f'body_pose_h{h1id}'],
            'transl_h1': item[f'transl_h{h1id}'],
            'translx_h1': item[f'transl_h{h1id}'][[0]],
            'transly_h1': item[f'transl_h{h1id}'][[1]],
            'translz_h1': item[f'transl_h{h1id}'][[2]],
            'betas_h1': item[f'betas_h{h1id}'][:10],
            'scale_h1': item[f'betas_h{h1id}'][[-1]],
            'joints_h1': item[f'joints_h{h1id}'],
            'vertices_h1': item[f'vertices_h{h1id}'],
            'bev_keypoints_h1': item[f'bev_keypoints_h{h1id}'],
            'bev_orig_vertices_h1': item[f'bev_orig_vertices_h{h1id}'],
            'op_keypoints_h1': item[f'op_keypoints_h{h1id}'],
            'vitpose_keypoints_h1': item[f'vitpose_keypoints_h{h1id}'],
            'keypoints_h1': item[f'vitpose_keypoints_h{h1id}'], # added it's the used keypoints
        }
                
        target = {**gen_target, **cam_target, **human_target}

        target = self.to_tensors(target)
        
        return target