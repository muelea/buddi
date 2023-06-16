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
        print(f'FlickrCI3D Dataset with child only: {child_only}, adult only: {adult_only}')
        
        # validation data/images must be loaded from training folder
        self.split_folder = 'test' if split == 'test' else 'train'
        if self.split_folder == 'train':
            trainval_fn = osp.join(self.processed_data_folder, 'train_val_split.npz')
            self.imgnames = np.load(trainval_fn)[self.split]

        self.body_model_type = body_model_type
        self.image_folder = osp.join(self.original_data_folder, self.split_folder, image_folder)
        self.openpose_folder = osp.join(self.processed_data_folder, self.split_folder, openpose_folder)
        self.bev_folder = osp.join(self.processed_data_folder, self.split_folder, bev_folder)
        self.vitpose_folder = osp.join(self.processed_data_folder, self.split_folder, vitpose_folder)
        self.pseudogt_folder = osp.join(self.processed_data_folder, self.split_folder, pseudogt_folder)
        self.has_pseudogt = False if pseudogt_folder == '' else True

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

    def bbox_from_openpose(self, op_data, kp_key='pose_keypoints_2d'):
        bbox = []
        for x in op_data:
            keypoints = x[kp_key]
            kpts = np.array(keypoints).reshape(-1,3)
            conf = kpts[:,-1]
            x0, y0, _ = kpts[conf > 0].min(0) # lower left corner
            x1, y1, _ = kpts[conf > 0].max(0) # upper right corner
            bbox.append([x0,y0,x1,y1])
        bbox = np.array(bbox)
        return bbox

    def bbox_from_bev(self, keypoints):
        llc = keypoints.min(1) # lower left corner
        urc = keypoints.max(1) # upper right corner
        bbox = np.hstack((llc, urc))
        return bbox

    def process_bev(self, bev_human_idx, bev_data, image_size):

            height, width = image_size

            # hacky - use smpl pose parameters with smplx body model
            # not perfect, but close enough. SMPL betas are not used with smpl-x.
            bev_betas = bev_data['smpl_betas'][bev_human_idx]
            age_val = bev_betas[-1]
            if self.body_model_type == 'smplx':
                body_pose = bev_data['smpl_thetas'][bev_human_idx][3:66]
                global_orient = bev_data['smpl_thetas'][bev_human_idx][:3]
                if age_val > 0.8:
                    betas = self.shape_converter_smil.forward(torch.from_numpy(bev_betas[:10]).unsqueeze(0)) # set to zero for smpl-x
                    return None # do not process infants
                else:
                    betas = self.shape_converter_smpla.forward(torch.from_numpy(bev_betas).unsqueeze(0)) # set to zero for smpl-x
                betas = betas[0].numpy()
            elif self.body_model_type == 'smpl':
                body_pose = bev_data['smpl_thetas'][bev_human_idx][3:]
                global_orient = bev_data['smpl_thetas'][bev_human_idx][:3]
                betas = bev_data['smpl_betas'][bev_human_idx][:10]
            
            # create smplxa 
            if False:
                model_folder = osp.join('../../../essentials', 'body_models')
                kid_template = osp.join(model_folder, 'smil/smplx_kid_template.npy')
                smplxa = smplx.create(
                    model_path=model_folder, 
                    model_type='smplx',
                    kid_template_path=kid_template, 
                    age='kid'
                )
                for age_val in np.linspace(0, 1, 11):
                    bev_betas[:] = 0
                    bev_betas[-1] = age_val
                    betas = self.shape_converter_smpla.forward(torch.from_numpy(bev_betas).unsqueeze(0)) # set to zero for smpl-x
                    betas = betas[0].numpy()
                    #betas[-1] = 1 - (1 / (1 + math.exp(-10.0 * betas[-1] + 5.0)))
                    sxbody = smplxa.forward_shape(
                        betas=torch.from_numpy(betas).unsqueeze(0),
                    )['v_shaped']
                    mm = trimesh.Trimesh(sxbody.detach().cpu().numpy()[0], self.body_model.faces, process=False)
                    _ = mm.export(f'stosx/smplxa/new_b0_check_agethres_{age_val}.ply')


            # visualize converted smpl-x betas
            #sbody = self.shape_converter.inbm.forward_shape(
            #    betas=torch.from_numpy(bev_betas).unsqueeze(0),
            #)['v_shaped']
            #sbody = sbody.detach().cpu().numpy()[0]
            # sf = self.shape_converter.inbm.faces
            #sbody = bev_data['verts'][bev_human_idx]
            #sf = self.shape_converter.inbm['f']
            #mm = trimesh.Trimesh(sbody, sf, process=False)
            #_ = mm.export(f'stosx/aa_{bev_human_idx}_smpl_mesh.ply')
            #sxbody = self.body_model.forward_shape(
            #    betas=torch.from_numpy(betas).unsqueeze(0),
            #)['v_shaped']
            #mm = trimesh.Trimesh(sxbody.detach().cpu().numpy()[0], self.body_model.faces, process=False)
            #_ = mm.export(f'stosx/check_agethres_{age_val}.ply')

            # Get the body translation. BEV root aligns meshed before rendering. 
            # we run SMPL to get the root joint
            # We also apply the camera translation to each mesh and use a single
            # camera instead
            bev_cam_trans = torch.from_numpy(bev_data['cam_trans'][bev_human_idx])
            body = self.body_model(
                global_orient=torch.from_numpy(global_orient).unsqueeze(0),
                body_pose=torch.from_numpy(body_pose).unsqueeze(0),
                betas=torch.from_numpy(betas).unsqueeze(0),
            )
            root_trans = body.joints.detach()[:,0,:]
            transl = -root_trans.to('cpu') + bev_cam_trans.to('cpu')
            transl = transl[0]

            body = self.body_model(
                global_orient=torch.from_numpy(global_orient).unsqueeze(0),
                body_pose=torch.from_numpy(body_pose).unsqueeze(0),
                betas=torch.from_numpy(betas).unsqueeze(0),
                transl=transl.unsqueeze(0),
            )
            joints = body.joints.detach().to('cpu').numpy()[0]
            vertices = body.vertices.detach().to('cpu').numpy()[0]

            # create bev camera 
            bev_camera = PerspectiveCamera(
                rotation=torch.tensor([[0., 0., 180.]]),
                translation=torch.tensor([[0., 0., 0.]]),
                afov_horizontal=torch.tensor([self.BEV_FOV]),
                image_size=torch.tensor([[width, height]]),
                batch_size=1,
                device='cpu'
            )
            keypoints = bev_camera.project(body.joints.detach())
            keypoints = keypoints.detach().numpy()[0]

            bev_joints3d = bev_data['joints'][bev_human_idx]
            bev_vertices = bev_data['verts'][bev_human_idx]
            bev_root_trans = bev_joints3d[[45,46]].mean(0)
            bev_vertices_root_trans = bev_vertices - bev_root_trans + bev_cam_trans.numpy()

            params = {
                'global_orient': global_orient,
                'body_pose': body_pose,
                'transl': transl,
                'betas': betas,
                'joints': joints,
                'vertices': vertices,
                'bev_keypoints': keypoints,
                'bev_orig_vertices': bev_vertices_root_trans,
                'bev_orig_betas': bev_data['smpl_betas'][bev_human_idx],
            }

            return params

    def load_single_image(self, imgname, annotation):
        
        # annotation / image paths
        img_path = osp.join(self.image_folder, f'{imgname}.png')
        bev_path = osp.join(self.bev_folder, f'{imgname}_0.08.npz')
        openpose_path = osp.join(self.openpose_folder, f'{imgname}.json')
        vitpose_path = osp.join(self.vitpose_folder, f'{imgname}_keypoints.json')

        # load each annotation file
        IMG = cv2.imread(img_path)
        bev_data = np.load(bev_path, allow_pickle=True)['results'][()]
        op_data = json.load(open(openpose_path, 'r'))['people']
        vitpose_data = json.load(open(vitpose_path, 'r'))['people']

        # smpl joints to openpose joints for BEV / Openpose matching
        smpl_to_op_map = smpl_to_openpose(model_type='smpl', use_hands=False, use_face=False,
                     use_face_contour=False, openpose_format='coco25')

        ################ Match HHC annotation with OpenPose and ViT keypoints #################
        op_bbox = self.bbox_from_openpose(op_data, kp_key='pose_keypoints_2d') 
        ci_bbox = np.array(annotation['bbxes'])
        ciop_iou_matrix, ciop_iou_best_match = iou_matrix(ci_bbox, op_bbox)

        opbev_kpcost_matrix, opbev_kpcost_best_match = keypoint_cost_matrix(
            kpts1=[np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in op_data],
            kpts2=[np.concatenate((x.reshape(-1,2)[smpl_to_op_map,:], np.ones((25, 1))), axis=1) for x in bev_data['pj2d_org']],
            norm=max(IMG.shape[0], IMG.shape[1])
        )
        opvitpose_kpcost_matrix, opvitpose_kpcost_best_match = keypoint_cost_matrix(
            kpts1=[np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in op_data],
            kpts2=[np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in vitpose_data],
            norm=max(IMG.shape[0], IMG.shape[1])
        )
   
        ################ camera params #################
        height, width, _ = IMG.shape
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

        outcost = []
        IMGS = []
        image_contact_data = []

        # load contact annotations
        for case_ci_idx, case_ci in enumerate(annotation['ci_sign']):
            IGNORE_PAIR = False
            image_data = image_data_template.copy()

            if self.has_pseudogt:
                pseudogt_path = osp.join(self.pseudogt_folder, f'{imgname}_{case_ci_idx}.pkl')
                with open(pseudogt_path, 'rb') as f:
                    pseudogt = pickle.load(f)

            image_data['contact_index'] = case_ci_idx
            # contact human id annotation
            person1_id, person2_id = case_ci['person_ids']
            image_data['hhc_contacts_human_ids'] = case_ci['person_ids']
            # contact regions annotation
            region_id = case_ci[self.body_model_type]['region_id']
            image_data['hhc_contacts_region_ids'] = region_id

            contact_map = self.contact_zeros.clone()
            for rid in region_id:
                contact_map[rid[0], rid[1]] = True
            image_data['contact_map'] = contact_map

            ################ load the two humans in contact #################
            human0_id = None
            for human_id, bbox_id in enumerate(case_ci['person_ids']):
                bbox = annotation['bbxes'][bbox_id]
                op_human_idx = ciop_iou_best_match[bbox_id]
                bev_human_idx = opbev_kpcost_best_match[op_human_idx]
                if human_id == 0: # ignore image if bounding box assignment is wrong (bev input estimated twice)
                    human0_id = bev_human_idx
                else:
                    if bev_human_idx == human0_id:
                        IGNORE_PAIR = True
                        break
                vitpose_human_idx = opvitpose_kpcost_best_match[op_human_idx]

                image_data[f'bbox_h{human_id}'] = bbox

                # add pseudo gt smpl-x params                
                # if bev detects child, do not use the image
                human_body_model_params = self.process_bev(bev_human_idx, bev_data, (height, width))
                if human_body_model_params is None:
                    IGNORE_PAIR = True
                    break
                for k, v in human_body_model_params.items():
                    image_data[f'{k}_h{human_id}'] = v

                # add keypoints openpose
                op_human_kpts = op_data[op_human_idx]
                image_data[f'op_keypoints_h{human_id}'] = np.array(op_human_kpts['pose_keypoints_2d']).reshape(-1,3)

                # add keypoints vitpose
                # ignore pair when not bev match was found
                if bev_human_idx == -1:
                    if self.split in ['train', 'val']:
                        IGNORE_PAIR = True
                        break
                    else:
                        print('At test time all images must be loaded. You must BEV match check.')
                        raise NotImplementedError

                # OpenPose and vit detection cost (if cost is too high, use Openpose)                
                if vitpose_human_idx == -1:
                    vitpose_human_kpts = op_human_kpts
                else:                    
                    detection_cost = opvitpose_kpcost_matrix[op_human_idx][vitpose_human_idx]
                    if detection_cost <= KEYPOINT_COST_TRHESHOLD:
                        vitpose_human_kpts = vitpose_data[vitpose_human_idx]
                    else:
                        vitpose_human_kpts = op_human_kpts
                image_data[f'vitpose_keypoints_h{human_id}'] = np.array(vitpose_human_kpts['pose_keypoints_2d']).reshape(-1,3)

                # add the pseudo grount-truth pose 
                if self.has_pseudogt:
                    for k, v in pseudogt[f'h{human_id}'].items():
                        image_data[f'pseudogt_{k}_h{human_id}'] = torch.from_numpy(v).float()[0]
            
            if not IGNORE_PAIR:
                image_contact_data.append(image_data)
            else:
                print('IGNORE PAIR', imgname, case_ci_idx)
                IGNORE_PAIR = False

       
        return image_contact_data

    def load(self):

        processed_data_path = osp.join(
            self.processed_data_folder, f'{self.split}.pkl'
        )

        # load data if it exists, otherwise process it
        if osp.exists(processed_data_path):      
            guru.info(f'Loading processed data from {processed_data_path}')   
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
        else:            

            guru.info(f'Processing data from {self.original_data_folder}')

            data = []
            # iterate though dataset / images
            for imgname, anno in tqdm(self.annotation.items()):
                # only lead images in split
                if self.split_folder == 'train' and imgname not in self.imgnames:
                    continue
               
                try:
                    data += self.load_single_image(imgname, anno) 
                except Exception as e:                
                    # if exeption is keyboard interrupt end program
                    if isinstance(e, KeyboardInterrupt):
                        raise e                    
                    else:
                        print(f'Error loading {imgname}')
                        print(f'Exception: {e}')
                        continue

            # save data to processed data folder
            with open(processed_data_path, 'wb') as f:
                pickle.dump(data, f)
        
        if self.child_only:
            data = [x for x in data if (x['bev_orig_betas_h0'][-1] > 0.2) or (x['bev_orig_betas_h1'][-1] > 0.2)]
        
        if self.adult_only:
            data = [x for x in data if (x['bev_orig_betas_h0'][-1] <= 0.2) and (x['bev_orig_betas_h1'][-1] <= 0.2)]

        if self.overfit:
            data = data[:self.overfit_num_samples]

        return data
        
        

