
from __future__ import division

import torch
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
import pickle
import json 

from llib.utils.image.augmentation import (
    crop, flip_img, flip_pose, flip_kp, transform, rot_aa
)
from llib.models.regressors.bev.utils import img_preprocess, bbox_preprocess

class SingleDataset(Dataset):
    def __init__(
             self,
             dataset_cfg,
             dataset_name,
             augmentation,
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
        super(SingleDataset, self).__init__()

        self.augmentation = augmentation
        self.image_processing = image_processing
        self.body_model_type = body_model_type
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.split = split 

        # contact regions to get contact-heat
        region_to_vertex = 'essentials/contact/flickrci3ds_r75_rid_to_smplx_vid.pkl'
        self.rid_to_vid = pickle.load(open(region_to_vertex, 'rb'))


        # for validation and test do not augment data
        if self.split != 'train':
            self.augmentation.use = False
        
        self.num_pose_params = 72
        self.num_shape_params = 10
        self.num_global_orient_params = 3
        self.num_transl_params = 3
        self.num_gt_kpts = 24
        self.num_op_kpts = 25

        self.IMGRES = self.image_processing.resolution

        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.body_model_type = body_model_type

        self.img_dir = osp.join(
            dataset_cfg.original_data_folder, dataset_cfg.image_folder
        )

        self.normalize_img = Normalize(
            mean=self.image_processing.normalization_mean,
            std=self.image_processing.normalization_std
        )

        self.data = self.load_data()
        self.len = len(self.data)

        # load action label dict 
        label_path = osp.join(
            dataset_cfg.processed_data_folder, 'action_to_class_id.json'
        )
        if osp.exists(label_path):
            self.action_label_dict = json.load(open(label_path, 'r'))
        else:
            self.action_label_dict = None

        self.set_feature_vec()

    def set_feature_vec(self):
        # create feature member variables
        feature_cfg = self.dataset_cfg.features
        feature_vec = np.ones(self.len).astype(bool)
        self.is_itw = feature_cfg.is_itw * feature_vec
        self.has_dhhc_class = feature_cfg.has_dhhc_class * feature_vec
        self.has_dhhc_sig = feature_cfg.has_dhhc_sig * feature_vec
        self.has_dsc_sig = feature_cfg.has_dsc_sig * feature_vec
        self.has_dsc_class = feature_cfg.has_dsc_class * feature_vec
        self.has_gt_kpts = feature_cfg.has_gt_kpts * feature_vec
        self.has_op_kpts = feature_cfg.has_op_kpts * feature_vec
        self.has_gt_joints = feature_cfg.has_gt_joints * feature_vec
        self.has_gt_smpl_shape = feature_cfg.has_gt_smpl_shape * feature_vec
        self.has_gt_smpl_pose = feature_cfg.has_gt_smpl_pose * feature_vec
        self.has_pgt_smpl_shape = feature_cfg.has_pgt_smpl_shape * feature_vec
        self.has_pgt_smpl_pose = feature_cfg.has_pgt_smpl_pose * feature_vec

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
                processed_fn_ext='_diffusion.pkl'  
            )
        elif self.dataset_name == 'chi3d':
            from .preprocess.chi3d import CHI3D
            dataset = CHI3D(
                **self.dataset_cfg, 
                split=self.split,
                body_model_type=self.body_model_type
            ).load(
                processed_fn_ext='_diffusion.pkl'  
            )
        elif self.dataset_name == 'hi4d':
            from .preprocess.hi4d import HI4D
            dataset = HI4D(
                **self.dataset_cfg, 
                split=self.split,
                body_model_type=self.body_model_type
            ).load(
                processed_fn_ext='_diffusion.pkl'  
            )
        else:
            raise NotImplementedError
            
        return dataset


    def _augm_params(self):
        """Get augmentation parameters."""

        mirror = 0         # mirror image
        pn = np.ones(3)    # per channel pixel-noise
        rot = 0            # rotation
        sc = 1             # scaling

        if self.split == 'train' and self.augmentation.use:
            # We flip with probability 1/2
            mirror_factor = self.augmentation.mirror
            if np.random.uniform() <= mirror_factor:
                mirror = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            noise_factor = self.augmentation.noise
            pn = np.random.uniform(1-noise_factor, 1+noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            # but zero with probability .6
            if np.random.uniform() > 0.6:
                rotation_factor = self.augmentation.rotation
                rot = min(2*rotation_factor,
                        max(-2*rotation_factor, np.random.randn()*rotation_factor)
                )

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            scale_factor = self.augmentation.scale_factor
            sc = min(1+scale_factor,
                    max(1-scale_factor, np.random.randn()*scale_factor+1))
            
        return mirror, pn, rot, sc

    def augm_params_threed(self):
        """Get augmentation parameters."""

        swap = 0

        if self.split == 'train' and self.augmentation.use:
            if np.random.uniform() < self.augmentation.swap:
                swap = 1
                
        return swap

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                      [self.IMGRES, self.IMGRES], rot=rot)
        # mirror the image
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))

        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):

        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale,
                                  [self.IMGRES, self.IMGRES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.IMGRES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        elif S.shape[1] == 3:
            S = np.einsum('ij,kj->ki', rot_mat, S)
        else:
            S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        pose = pose.astype('float32')
        return pose

    def get_single_item_features(self, index):
        features = {
            'is_itw': self.is_itw[index],
            'has_dhhc_class': self.has_dhhc_class[index],
            'has_dhhc_sig': self.has_dhhc_sig[index],
            'has_dsc_sig': self.has_dsc_sig[index],
            'has_dsc_class': self.has_dsc_class[index],
            'has_gt_kpts': self.has_gt_kpts[index],
            'has_op_kpts': self.has_op_kpts[index],
            'has_gt_joints': self.has_gt_joints[index],
            'has_gt_smpl_shape': self.has_gt_smpl_shape[index],
            'has_gt_smpl_pose': self.has_gt_smpl_pose[index],
            'has_pgt_smpl_shape': self.has_pgt_smpl_shape[index],
            'has_pgt_smpl_pose': self.has_pgt_smpl_pose[index],

        } 
        return features

    def join_bbox(self, bb1, bb2):
        x1, y1 = min(bb1[0], bb2[0]), min(bb1[1], bb2[1])
        x2, y2 = max(bb1[2], bb2[2]), max(bb1[3], bb2[3])
        return [x1, y1, x2, y2]

    def visu_item(self, orig_img, h1_bbox, h2_bbox, h1_joints, h2_joints):

        import cv2
        
        IMG = orig_img
        h1_bbox = [int(x) for x in h1_bbox]
        h2_bbox = [int(x) for x in h2_bbox]
        h12_bbox = self.join_bbox(h1_bbox, h2_bbox)

        h1_joints = np.array(h1_joints).astype(int)
        h2_joints = np.array(h1_joints).astype(int)

        h1_col = (0, 255, 0)
        h2_col = (0, 0, 255)
        h12_col = (0, 255, 255)
        # add bounding box to input image 
        IMG = cv2.rectangle(IMG, (h1_bbox[0], h1_bbox[1]), (h1_bbox[2], h1_bbox[3]), h1_col, 2)
        IMG = cv2.rectangle(IMG, (h2_bbox[0], h2_bbox[1]), (h2_bbox[2], h2_bbox[3]), h2_col, 2)
        IMG = cv2.rectangle(IMG, (h12_bbox[0], h12_bbox[1]), (h12_bbox[2], h12_bbox[3]), h12_col, 2)

        # add joints to the input image
        for idx, joint in enumerate(h1_joints):
            IMG = cv2.circle(IMG, (int(joint[0]), int(joint[1])), 3, h1_col, 2)
        for idx, joint in enumerate(h2_joints):
            IMG = cv2.circle(IMG, (int(joint[0]), int(joint[1])), 3, h2_col, 2)

        cv2.imwrite('check_annotation.png', IMG[:,:,[2,1,0]])
        
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
        ds_features = self.get_single_item_features(index)
        
        #h0id = 0 if np.array(item['bbox_h0'])[[1,3]].mean() <= np.array(item['bbox_h1'])[[1,3]].mean() else 1
        #h1id = 1-h0id

        # crop image using both bounding boxes
        #h1_bbox, h2_bbox = item[f'bbox_h0'], item[f'bbox_h1']
        #if 'flickr_bbox' in item.keys():
        #    bbox = item['flickr_bbox'].astype(int)
        #else:
        #    import ipdb; ipdb.set_trace()

        # Load image and resize directly before cropping, because of speed
        if self.image_processing.load_image:
            bbox_join = self.join_bbox(bbox[0], bbox[1])
            bbox_join = np.array(bbox_join).astype(int)
            orig_img = cv2.imread(item['imgpath'])
            height, width = orig_img.shape[:2]

            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(width-1, bbox[2])
            bbox[3] = min(height-1, bbox[3])
            cropped_image = orig_img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            xmin = bbox[1].copy()
            ymin = bbox[0].copy()
            bbox[[1,3]] -= xmin
            bbox[[0,2]] -= ymin
            h1_bbox[[1,3]] -= xmin
            h1_bbox[[0,2]] -= ymin 
            h2_bbox[[1,3]] -= xmin
            h2_bbox[[0,2]] -= ymin
            
            input_image, image_pad_info, pad_image = img_preprocess(
                cropped_image.copy(), input_size=512, return_pad_img=True)
            input_image = input_image[0]
            # add padding to bounding box paramss
            h1_bbox = bbox_preprocess(h1_bbox, image_pad_info, pad_image, image_size=512)
            h2_bbox = bbox_preprocess(h2_bbox, image_pad_info, pad_image, image_size=512)
            bbox = bbox_preprocess(bbox, image_pad_info, pad_image, image_size=512)

            """
            cv2.imwrite(f'outdebug/{index}.png', orig_img)
            """ 
            img_height, img_width, _ = input_image.shape            
   
        else:
            input_image = [0.0]
            img_height = item['img_height']
            img_width = item['img_width']

        # Get augmentation parameters and process image
        #flip, pn, rot, sc = self.augm_params()     
        swap = self.augm_params_threed()

        if 'contact_index' in item.keys():
            contact_index = item['contact_index'] if item['contact_index'] is not None else 0
        else:
            contact_index = 0

        # get action label 
        if self.action_label_dict is not None:
            action_name = item['imgname'].split('_')[0].split(' ')[0].lower()
            action = self.action_label_dict[action_name]
        else:
            action_name = ''
            action = -1

        img_out_fn = item['imgname'].replace('.png', '_') + str(contact_index)
        gen_target = {
            'images': input_image,
            'imgpath': item['imgpath'],
            'action_name': action_name,
            'action': action,
            'contact_index': contact_index,
            'imgname_fn': item['imgname'],
            'imgname_fn_out': img_out_fn,
            'img_height': img_height,
            'img_width': img_width,
            'sample_index': index,
        }

        cam_target = {}
        if 'cam_rot' in item.keys():
            if item['cam_rot'] is not None:
                cam_target = {
                    'cam_rot': np.array(item['cam_rot']),
                    'cam_transl': np.array(item['cam_transl']),
                    'fl': np.array(item['fl']),
                    'ih': np.array(item['img_height']),
                    'iw': np.array(item['img_width']),
                }


        if 'contact_map' in item.keys():
            contact_map = item['contact_map']
        else:
            contact_map = np.zeros((75, 75)).astype(bool)
    

        human_target = {}
        # first order them with respect to the x coordinate
        # and swap if augmentation is set to true
        #if 'pgt_smplx_transl' in item.keys():
        #    transl_param = item['pgt_smplx_transl']
        #elif 'bev_smplx_transl' in item.keys():
        #    transl_param = item['bev_smplx_transl']
        #elif 'pgt_transl' in item.keys():
        #    transl_param = item['pgt_transl']
        #elif 'transl_smpl' in item.keys():
        #    transl_param = item['transl_smpl']
        #elif 'transl' in item.keys():
        #    transl_param = item['transl'][:,0,:]
        #else:
        #    raise ValueError('No pgt_smplx_transl in item keys')

        if 'FlickrCI3D' in item['imgpath']:
            transl_param = item[f'pgt_{self.body_model_type}_transl']
        elif 'CHI3D' in item['imgpath']:
            transl_param = item['transl'].squeeze(1)
        elif 'Hi4D' in item['imgpath']:
            transl_param = item[f'transl_{self.body_model_type}']
        else:
            raise NotImplementedError

        h0id = 0 if transl_param[0][0] <= transl_param[1][0] else 1
        h1id = 1-h0id
        if swap:
            h0id, h1id = h1id, h0id # augment if set to true
        idxs = [h0id, h1id]
        if h0id == 1:
            contact_map = contact_map.T
            #gen_target['bbox'] = gen_target['bbox'][idxs]

        # prefix and body model type to get correct key
        if 'FlickrCI3D' in item['imgpath']:
            human_target = {
                'contact_map': contact_map, 
                'pgt_global_orient': item[f'pgt_{self.body_model_type}_global_orient'][idxs],
                'pgt_body_pose': item[f'pgt_{self.body_model_type}_body_pose'][idxs],
                'pgt_transl': item[f'pgt_{self.body_model_type}_transl'][idxs],
                'pgt_betas': item[f'pgt_{self.body_model_type}_betas'][idxs],
                'pgt_scale': item[f'pgt_{self.body_model_type}_scale'][idxs], 
                'bev_global_orient': item[f'bev_{self.body_model_type}_global_orient'][idxs],
                'bev_body_pose': item[f'bev_{self.body_model_type}_body_pose'][idxs],
                'bev_transl': item[f'bev_{self.body_model_type}_transl'][idxs],
                'bev_betas': item[f'bev_{self.body_model_type}_betas'][idxs],
                'bev_scale': item[f'bev_{self.body_model_type}_scale'][idxs][:,None],
            }
        elif 'CHI3D' in item['imgpath']:
            human_target = {
                'contact_map': contact_map, 
                'pgt_global_orient': item[f'global_orient_cam'][idxs,0],
                'pgt_body_pose': item[f'body_pose'][idxs,0],
                'pgt_transl': item[f'transl_cam'][idxs,0],
                'pgt_betas': item[f'betas'][idxs,0],
                'pgt_scale': item[f'scale'][idxs,0],
                'bev_global_orient': item[f'bev_{self.body_model_type}_global_orient'][idxs],
                'bev_body_pose': item[f'bev_{self.body_model_type}_body_pose'][idxs],
                'bev_transl': item[f'bev_{self.body_model_type}_transl'][idxs],
                'bev_betas': item[f'bev_{self.body_model_type}_betas'][idxs],
                'bev_scale': item[f'bev_{self.body_model_type}_scale'][idxs][:,None],
            }
            if self.dataset_cfg.load_unit_glob_and_transl:
                human_target['pgt_global_orient'] = item[f'global_orient'][idxs,0]
                human_target['pgt_transl'] = item['transl'][idxs,0]

        elif 'Hi4D' in item['imgpath']:
            human_target = {
                'contact_map': contact_map, 
                # 'pgt_global_orient': item[f'global_orient_{self.body_model_type}'][idxs].astype(np.float32),
                'pgt_global_orient': item[f'global_orient_cam_{self.body_model_type}'][idxs].astype(np.float32),
                'pgt_body_pose': item[f'body_pose_{self.body_model_type}'][idxs].astype(np.float32),
                # 'pgt_transl': item[f'transl_{self.body_model_type}'][idxs].astype(np.float32),
                'pgt_transl': item[f'transl_cam_{self.body_model_type}'][idxs].astype(np.float32),
                'pgt_betas': item[f'betas_{self.body_model_type}'][idxs].astype(np.float32),
                'pgt_scale': np.zeros((2,1))[idxs].astype(np.float32),
                'bev_global_orient': item[f'bev_{self.body_model_type}_global_orient'][idxs],
                'bev_body_pose': item[f'bev_{self.body_model_type}_body_pose'][idxs],
                'bev_transl': item[f'bev_{self.body_model_type}_transl'][idxs],
                'bev_betas': item[f'bev_{self.body_model_type}_betas'][idxs],
                'bev_scale': item[f'bev_{self.body_model_type}_scale'][idxs][:,None],
            }

            if self.dataset_cfg.load_unit_glob_and_transl:
                human_target['pgt_global_orient'] = item[f'global_orient_{self.body_model_type}'][idxs].astype(np.float32)
                human_target['pgt_transl'] = item[f'transl_{self.body_model_type}'][idxs].astype(np.float32)

        else:
            raise NotImplementedError
                

        target = {**gen_target, **cam_target, **human_target, **ds_features}

        target = self.to_tensors(target)

        keys = [x for x in target.keys()]
        for k in keys:
            if target[k] is None:
                target.pop(k) #[k] = target[k].to(self.device)

        return target
