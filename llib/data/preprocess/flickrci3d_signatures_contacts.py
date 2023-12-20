import os.path as osp 
import json 
import torch
import numpy as np
import os
import cv2
import smplx
import math
import pickle
import trimesh
from tqdm import tqdm
from llib.utils.image.bbox import iou_matrix
from llib.utils.keypoints.matching import keypoint_cost_matrix
from llib.defaults.body_model.main import conf as body_model_conf
from llib.cameras.perspective import PerspectiveCamera
from llib.bodymodels.utils import smpl_to_openpose
from loguru import logger as guru
from llib.data.preprocess.utils.shape_converter import ShapeConverter
        
KEYPOINT_COST_TRHESHOLD = 0.008

import torch
import torch.nn as nn


class FlickrCI3D_Signatures():
    
    BEV_FOV = 60

    def __init__(
        self,
        original_data_folder,
        processed_data_folder,
        imar_vision_datasets_tools_folder,
        image_folder,
        bev_folder='bev',
        openpose_folder='openpose',
        split='train',
        body_model_type='smplx',
        vitpose_folder='vitpose',
        vitposeplus_folder='vitposeplus',
        pseudogt_folder='pseudo_gt',
        max_count_regions_in_contact=26,
        number_of_regions=75, 
        overfit=False,
        overfit_num_samples=12,
        child_only=False,
        adult_only=False,
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder
        self.imar_vision_datasets_tools_folder =  imar_vision_datasets_tools_folder
        self.split = split
        self.child_only = child_only
        self.adult_only = adult_only
        #print(f'FlickrCI3D Dataset with child only: {child_only}, adult only: {adult_only}')
        
        # validation data/images must be loaded from training folder
        self.split_folder = 'test' if split == 'test' else 'train'
        if self.split_folder == 'train':
            trainval_fn = osp.join(self.processed_data_folder, 'train', 'train_val_split.npz')
            self.imgnames = np.load(trainval_fn)[self.split]

        self.body_model_type = body_model_type
        self.image_folder = osp.join(self.original_data_folder, self.split_folder, image_folder)
        self.openpose_folder = osp.join(self.processed_data_folder, self.split_folder, openpose_folder)
        self.bev_folder = osp.join(self.processed_data_folder, self.split_folder, bev_folder)
        self.vitpose_folder = osp.join(self.processed_data_folder, self.split_folder, vitpose_folder)
        self.vitposeplus_folder = osp.join(self.processed_data_folder, self.split_folder, vitposeplus_folder)
        self.pseudogt_folder = osp.join(self.processed_data_folder, self.split_folder, pseudogt_folder)
        self.has_pseudogt = False if pseudogt_folder == '' else True

        if self.has_pseudogt:
            self.pseudogt_fits = pickle.load(
                open(osp.join(self.processed_data_folder, self.split_folder, 'processed_pseudogt_fits.pkl'), 'rb')
            )

        # load processed data (bev, keypoints, etc.)
        processed_fn = osp.join(
            self.processed_data_folder, self.split_folder, 'processed.pkl'
        )
        self.processed = pickle.load(open(processed_fn, 'rb'))

        annotation_fn = osp.join(
            self.original_data_folder, self.split_folder, 'interaction_contact_signature.json'
        )
        self.annotation = json.load(open(annotation_fn, 'r'))


        contact_regions_fn = osp.join(
            self.imar_vision_datasets_tools_folder, 'info/contact_regions.json'
        )
        contact_regions = json.load(open(contact_regions_fn, 'r'))
        self.rid_to_smplx_fids = contact_regions['rid_to_smplx_fids']

        # Get SMPL-X pose, if available
        self.global_orient = torch.zeros(3, dtype=torch.float32)
        self.body_pose = torch.zeros(63, dtype=torch.float32)
        self.betas = torch.zeros(10, dtype=torch.float32)
        self.transl = torch.zeros(3, dtype=torch.float32)

        # keypoints 
        self.keypoints = torch.zeros((24, 3), dtype=torch.float32)

        self.num_verts = 10475 if self.body_model_type == 'smplx' else 6890
        self.max_count_regions_in_contact = max_count_regions_in_contact
        self.number_of_regions = number_of_regions
        self.contact_zeros = torch.zeros(
            (self.number_of_regions, self.number_of_regions)
        ).to(torch.bool)

        # convert smpl betas to smpl-x betas 
        self.shape_converter_smpla = ShapeConverter(inbm_type='smpla', outbm_type='smplxa')
        self.shape_converter_smil = ShapeConverter(inbm_type='smil', outbm_type='smplxa')

        # create body model to get bev root translation from pose params
        self.body_model = self.shape_converter_smpla.outbm

        # for overfitting experiments we use the first 12 samples
        self.overfit = overfit
        self.overfit_num_samples = overfit_num_samples

    def process_bev(self, data, image_size):

            height, width = image_size

            # hacky - use smpl pose parameters with smplx body model
            # not perfect, but close enough. SMPL betas are not used with smpl-x.
            if self.body_model_type == 'smplx':
                body_pose = data['bev_smpl_body_pose'][:,:63]
                global_orient = data['bev_smpl_global_orient']
                betas = data['bev_smplx_betas']
                scale = data['bev_smplx_scale']
            else:
                raise('not implemented: Data loader for SMPL loader in Flickr Signatures')

            # ignore infants, because SMPL-X doesn't support them (template is noisy)
            has_infant = False
            if np.any(data['bev_smpl_scale'] > 0.8):
                has_infant = True
            
            bev_cam_trans = torch.from_numpy(data['bev_cam_trans'])
            bev_camera = PerspectiveCamera(
                rotation=torch.tensor([[0., 0., 180.]]),
                translation=torch.tensor([[0., 0., 0.]]),
                afov_horizontal=torch.tensor([self.BEV_FOV]),
                image_size=torch.tensor([[width, height]]),
                batch_size=1,
                device='cpu'
            )
            bev_vertices = data['bev_smpl_vertices']
            bev_root_trans = data['bev_smpl_joints'][:,[45,46],:].mean(1)
            bev_vertices_root_trans = bev_vertices - bev_root_trans[:,np.newaxis,:] \
                + bev_cam_trans.numpy()[:,np.newaxis,:]
            data['bev_smpl_vertices_root_trans'] = bev_vertices_root_trans
            
            smplx_update = {
                'bev_smplx_global_orient': [],
                'bev_smplx_body_pose': [],
                'bev_smplx_transl': [],
                'bev_smplx_keypoints': [],
                'bev_smplx_vertices': [],
            }

            for idx in range(2):

                h_global_orient = torch.from_numpy(global_orient[[idx]]).float()
                smplx_update['bev_smplx_global_orient'].append(h_global_orient)
                
                h_body_pose = torch.from_numpy(body_pose[[idx]]).float()
                smplx_update['bev_smplx_body_pose'].append(h_body_pose)

                h_betas_scale = torch.from_numpy(
                    np.concatenate((betas[[idx]], scale[[idx]][None]), axis=1)
                ).float()

                body = self.body_model(
                    global_orient=h_global_orient,
                    body_pose=h_body_pose,
                    betas=h_betas_scale,
                )

                root_trans = body.joints.detach()[:,0,:]
                transl = -root_trans.to('cpu') + bev_cam_trans[[idx]].to('cpu')
                smplx_update['bev_smplx_transl'].append(transl)

                body = self.body_model(
                    global_orient=h_global_orient,
                    body_pose=h_body_pose,
                    betas=h_betas_scale,
                    transl=transl,
                )

                keypoints = bev_camera.project(body.joints.detach())
                smplx_update['bev_smplx_keypoints'].append(keypoints)

                vertices = body.vertices.detach().to('cpu')
                smplx_update['bev_smplx_vertices'].append(vertices)

            for k, v in smplx_update.items():
                smplx_update[k] = torch.cat(v, dim=0)

            data.update(smplx_update)

            return data, has_infant

    def concatenate_dicts(self, x, y):
        concatenated_dict = {}
        for key in x.keys():
            concatenated_dict[key] = np.stack((x[key], y[key]), axis=0)
        return concatenated_dict

    def load_single_image(self, imgname, annotation):

        processed = self.processed[imgname]
        
        # annotation / image paths
        img_path = osp.join(self.image_folder, f'{imgname}.png')

        ################ camera params #################
        height, width = processed['img_height'], processed['img_width']
        # camera translation was already applied to mesh, so we can set it to zero.
        cam_transl = [0., 0., 0.] 
        # camera rotation needs 180 degree rotation around z axis, because bev and
        # pytorch3d camera coordinate systems are different            
        cam_rot = [0., 0., 180.]
        afov_radians = (self.BEV_FOV / 2) * math.pi / 180
        focal_length_px = (max(width, height)/2) / math.tan(afov_radians)

        image_data_template = {
            'imgname': f'{imgname}.png',
            'imgpath': img_path,
            'img_height': height,
            'img_width': width,
            'cam_transl': cam_transl,
            'cam_rot': cam_rot,
            'fl': focal_length_px,
            'afov_horizontal': self.BEV_FOV,
            }

        # load contact annotations
        all_image_contact_data = []
        for case_ci_idx, case_ci in enumerate(annotation['ci_sign']):
            p0id, p1id = case_ci['person_ids']
            
            image_data = image_data_template.copy()

            image_data['contact_index'] = case_ci_idx
            image_data['hhc_contacts_human_ids'] = case_ci['person_ids']
            region_id = case_ci[self.body_model_type]['region_id']
            image_data['hhc_contacts_region_ids'] = region_id

            contact_map = self.contact_zeros.clone()
            for rid in region_id:
                contact_map[rid[0], rid[1]] = True
            image_data['contact_map'] = contact_map

            ################ load the two humans in contact #################
            human_data = self.concatenate_dicts(processed[p0id], processed[p1id])
            # update with bev params
            human_data, has_infant = self.process_bev(human_data, (processed['img_height'], processed['img_width']))
            image_data['has_infant'] = has_infant
            image_data.update(human_data)

            # check if all detections (keypoints, bev) are available 
            image_data['information_missing'] = False
            for x in ['openpose_human_idx', 'bev_human_idx', 'vitpose_human_idx', 'vitposeplus_human_idx']:
                if np.any(human_data[x] == -1):
                    image_data['information_missing'] = True

            # check if bev is missing or if the same person was selected for initializations
            image_data['bev_missing'] = False
            if np.unique(human_data['bev_human_idx']).shape[0] == 1 or np.any(human_data['bev_human_idx'] == -1):
                image_data['bev_missing'] = True
     

            # load pseudo gt smpl-x params
            if self.has_pseudogt:
                if f'{imgname}_{case_ci_idx}' in self.pseudogt_fits.keys():
                    pseudogt = self.pseudogt_fits[f'{imgname}_{case_ci_idx}']
                    human_pgt_data = {
                        'pgt_smplx_betas': pseudogt['humans']['betas'],
                        'pgt_smplx_global_orient': pseudogt['humans']['global_orient'],
                        'pgt_smplx_body_pose': pseudogt['humans']['body_pose'],
                        'pgt_smplx_transl': pseudogt['humans']['transl'],
                        'pgt_smplx_scale': pseudogt['humans']['scale'],
                    }         

                    image_data.update(human_pgt_data)
                else:
                    guru.warning(f'Pseudo GT not found for {imgname}_{case_ci_idx}. Skipping.')
                    return []
            
            all_image_contact_data.append(image_data)

        return all_image_contact_data

    def load(self, load_from_scratch=False, allow_missing_information=True, processed_fn_ext='.pkl'):
        """
        Load the dataset.
        ------------------ 
        load_from_scatch: if True, process the data from scratch, otherwise load pkl file.
        """
        processed_data_path = osp.join(
            self.processed_data_folder, f'{self.split}{processed_fn_ext}'
        )
        processed_data_path_exists = osp.exists(processed_data_path) 
        guru.info(f'Processed data path {processed_data_path} exists: {processed_data_path_exists}')
        guru.info(f'Load from scratch: {load_from_scratch}')

        # load data if it exists, otherwise process it
        if processed_data_path_exists and not load_from_scratch:   
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
            num_samples = len(data)
            guru.info(f'Loading processed data from {processed_data_path}. Num samples {num_samples}.')   
        else:            

            guru.info(f'Processing data from {self.original_data_folder}')

            data = []
            # iterate though dataset / images
            for imgname, anno in tqdm(self.annotation.items()):
                # only read images in split for train/val
                if self.split_folder == 'train' and imgname not in self.imgnames:
                    continue
                
                data += self.load_single_image(imgname, anno)

            # save data to processed data folder
            with open(processed_data_path, 'wb') as f:
                pickle.dump(data, f)
        
        #if self.child_only:
        #    data = [x for x in data if (x['bev_orig_betas_h0'][-1] > 0.2) or (x['bev_orig_betas_h1'][-1] > 0.2)]
        
        #if self.adult_only:
        #    data = [x for x in data if (x['bev_orig_betas_h0'][-1] <= 0.2) and (x['bev_orig_betas_h1'][-1] <= 0.2)]


        if not allow_missing_information:
            data = [x for x in data if not x['bev_missing']]
            data = [x for x in data if not x['has_infant']]
        
        if self.overfit:
            data = data[:self.overfit_num_samples]

        guru.info(f'Final number of samples in flickrci3ds: {len(data)}')
        
        return data
        
        

