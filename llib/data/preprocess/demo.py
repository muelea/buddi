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
import sys
from tqdm import tqdm
from llib.utils.image.bbox import iou_matrix
from llib.utils.keypoints.matching import keypoint_cost_matrix
from llib.defaults.body_model.main import conf as body_model_conf
from llib.cameras.perspective import PerspectiveCamera
from llib.bodymodels.utils import smpl_to_openpose
from loguru import logger as guru
from llib.data.preprocess.utils.shape_converter import ShapeConverter
from llib.utils.image.bbox import iou
        

import torch
import torch.nn as nn

KEYPOINT_COST_TRHESHOLD = 0.008

def check_bev_estimate(human_id, human0_id, bev_human_idx):
    """ 
    Check if the best match in BEV is two different people
    for human 0 and human 1.
    """
    ignore = False
    if human_id == 0:
        # first detected person, save detected bev index
        human0_id = bev_human_idx
    else:
        # second detected person, check if detected bev index is the same
        # as for first detected person. If so, ignore image.
        if bev_human_idx == human0_id:
            ignore = True
    return human0_id, ignore

def compare_and_select_openpose_vitpose(
    vitpose_data, op_human_kpts, opvitpose_kpcost_matrix,
    op_human_idx, vitpose_human_idx, KEYPOINT_COST_TRHESHOLD
):
    if vitpose_human_idx == -1:
        vitpose_human_kpts = op_human_kpts
    else:                    
        detection_cost = opvitpose_kpcost_matrix[op_human_idx][vitpose_human_idx]
        if detection_cost <= KEYPOINT_COST_TRHESHOLD:
            vitpose_human_kpts = vitpose_data[vitpose_human_idx]
        else:
            vitpose_human_kpts = op_human_kpts
    return vitpose_human_kpts

class Demo():
    
    BEV_FOV = 60

    def __init__(
        self,
        original_data_folder,
        image_folder='images',
        bev_folder='bev',
        openpose_folder='openpose',
        vitpose_folder='vitpose',
        number_of_regions=75, 
        imar_vision_datasets_tools_folder=None,
        has_gt_contact_annotation=False,
        image_format='png',
        image_name_select='',
        unique_keypoint_match=True,
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.data_folder = original_data_folder
        self.image_format = image_format
        self.image_name_select = image_name_select
        self.image_folder = osp.join(self.data_folder, image_folder)
        self.openpose_folder = osp.join(self.data_folder, openpose_folder)
        self.bev_folder = osp.join(self.data_folder, bev_folder)
        self.vitpose_folder = osp.join(self.data_folder, vitpose_folder)

        # convert smpl betas to smpl-x betas 
        self.shape_converter_smpla = ShapeConverter(inbm_type='smpla', outbm_type='smplxa')
        self.shape_converter_smil = ShapeConverter(inbm_type='smil', outbm_type='smplxa')

        # create body model to get bev root translation from pose params
        self.body_model = self.shape_converter_smpla.outbm
        self.body_model_type = 'smplx' # read smplx and convert to smplxa

        # load contact annotations is available
        self.has_gt_contact_annotation = has_gt_contact_annotation
        # if self.has_gt_contact_annotation:
        #     self.imar_vision_datasets_tools_folder =  imar_vision_datasets_tools_folder
        #     annotation_fn = osp.join(
        #         self.data_folder, 'interaction_contact_signature.json'
        #     )
        #     if os.path.exists(annotation_fn):
        #         self.annotation = json.load(open(annotation_fn, 'r'))


        #     contact_regions_fn = osp.join(
        #         self.imar_vision_datasets_tools_folder, 'info/contact_regions.json'
        #     )
        #     contact_regions = json.load(open(contact_regions_fn, 'r'))
        #     self.rid_to_smplx_fids = contact_regions['rid_to_smplx_fids']

        self.number_of_regions = number_of_regions
        self.contact_zeros = torch.zeros(
            (self.number_of_regions, self.number_of_regions)
        ).to(torch.bool)

        # Get SMPL-X pose, if available
        self.global_orient = torch.zeros(3, dtype=torch.float32)
        self.body_pose = torch.zeros(63, dtype=torch.float32)
        self.betas = torch.zeros(10, dtype=torch.float32)
        self.transl = torch.zeros(3, dtype=torch.float32)

        # keypoints 
        self.keypoints = torch.zeros((24, 3), dtype=torch.float32)
        self.unique_keypoint_match = unique_keypoint_match

        # smpl joints to openpose joints for BEV / Openpose matching
        self.smpl_to_op_map = smpl_to_openpose(
            model_type='smpl', use_hands=False, use_face=False,
            use_face_contour=False, openpose_format='coco25')

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

        smpl_betas_scale = bev_data['smpl_betas'][bev_human_idx]
        smpl_betas = smpl_betas_scale[:10]
        smpl_scale = smpl_betas_scale[-1]
        smpl_body_pose = bev_data['smpl_thetas'][bev_human_idx][3:]
        smpl_global_orient = bev_data['smpl_thetas'][bev_human_idx][:3]

        if smpl_scale > 0.8:
            smplx_betas_scale = self.shape_converter_smil.forward(torch.from_numpy(smpl_betas).unsqueeze(0))
            smplx_betas = smplx_betas_scale[0,:10].numpy()
            smplx_scale = smplx_betas_scale[0,10].numpy()
            #smplx_scale = smpl_scale # there is no smilxa model, so we keep the scale form bev
        else:
            smplx_betas_scale = self.shape_converter_smpla.forward(torch.from_numpy(smpl_betas_scale).unsqueeze(0))
            smplx_betas = smplx_betas_scale[0,:10].numpy()
            smplx_scale = smplx_betas_scale[0,10].numpy()

        cam_trans = bev_data['cam_trans'][bev_human_idx]
        smpl_joints = bev_data['joints'][bev_human_idx]
        smpl_vertices = bev_data['verts'][bev_human_idx]
        smpl_joints_2d = bev_data['pj2d_org'][bev_human_idx]

        data = {
            'bev_smpl_global_orient': smpl_global_orient,
            'bev_smpl_body_pose': smpl_body_pose,
            'bev_smpl_betas': smpl_betas,
            'bev_smpl_scale': smpl_scale,
            'bev_smplx_betas': smplx_betas,
            'bev_smplx_scale': smplx_scale,
            'bev_cam_trans': cam_trans,
            'bev_smpl_joints': smpl_joints,
            'bev_smpl_vertices': smpl_vertices,
            'bev_smpl_joints_2d': smpl_joints_2d,
        }
        
        height, width = image_size

        # hacky - use smpl pose parameters with smplx body model
        # not perfect, but close enough. SMPL betas are not used with smpl-x.
        if self.body_model_type == 'smplx':
            body_pose = data['bev_smpl_body_pose'][:63]
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
        bev_root_trans = data['bev_smpl_joints'][[45,46],:].mean(0)
        bev_vertices_root_trans = bev_vertices - bev_root_trans[np.newaxis,:] \
            + bev_cam_trans.numpy()[np.newaxis,:]
        data['bev_smpl_vertices_root_trans'] = bev_vertices_root_trans
        
        smplx_update = {
            'bev_smplx_global_orient': [],
            'bev_smplx_body_pose': [],
            'bev_smplx_transl': [],
            'bev_smplx_keypoints': [],
            'bev_smplx_vertices': [],
        }

        idx = 0
        h_global_orient = torch.from_numpy(global_orient).float().unsqueeze(0)
        smplx_update['bev_smplx_global_orient'].append(h_global_orient)
        
        h_body_pose = torch.from_numpy(body_pose).float().unsqueeze(0)
        smplx_update['bev_smplx_body_pose'].append(h_body_pose)

        h_betas_scale = torch.from_numpy(
            np.concatenate((betas, scale[None]), axis=0)
        ).float().unsqueeze(0)

        body = self.body_model(
            global_orient=h_global_orient,
            body_pose=h_body_pose,
            betas=h_betas_scale,
        )

        root_trans = body.joints.detach()[:,0,:]
        transl = -root_trans.to('cpu') + bev_cam_trans.to('cpu')
        smplx_update['bev_smplx_transl'].append(transl)

        body = self.body_model(
            global_orient=h_global_orient,
            body_pose=h_body_pose,
            betas=h_betas_scale,
            transl=transl,
        )

        keypoints = bev_camera.project(body.joints.detach())
        smplx_update['bev_smplx_keypoints'].append(keypoints.detach())

        vertices = body.vertices.detach().to('cpu')
        smplx_update['bev_smplx_vertices'].append(vertices)

        for k, v in smplx_update.items():
            smplx_update[k] = torch.cat(v, dim=0)

        data.update(smplx_update)

        return data, has_infant

    def read_data(self, imgname):

        # annotation / image paths
        img_path = osp.join(self.image_folder, f'{imgname}.{self.image_format}')
        bev_path = osp.join(self.bev_folder, f'{imgname}_0.08.npz')
        openpose_path = osp.join(self.openpose_folder, f'{imgname}.json')
        vitpose_path = osp.join(self.vitpose_folder, f'{imgname}_keypoints.json')

        # load each annotation file
        IMG = cv2.imread(img_path)
        bev_data = np.load(bev_path, allow_pickle=True)['results'][()]
        op_data = json.load(open(openpose_path, 'r'))['people']
        vitpose_data = json.load(open(vitpose_path, 'r'))['people']

        return img_path, IMG, bev_data, op_data, vitpose_data   

    def _get_opbev_cost(self, op_data, bev_data, IMG, unique_best_matches=True):
        matrix, best_match = keypoint_cost_matrix(
            kpts1=[np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in op_data],
            kpts2=[np.concatenate((x.reshape(-1,2)[self.smpl_to_op_map,:], np.ones((25, 1))), axis=1) for x in bev_data['pj2d_org']],
            norm=max(IMG.shape[0], IMG.shape[1]),
            unique_best_matches=unique_best_matches
        )
        return matrix, best_match

    def _get_opvitpose_cost(self, op_data, vitpose_data, IMG, unique_best_matches=True):
        matrix, best_match = keypoint_cost_matrix(
            kpts1=[np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in op_data],
            kpts2=[np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in vitpose_data],
            norm=max(IMG.shape[0], IMG.shape[1]),
            unique_best_matches=unique_best_matches
        )
        return matrix, best_match

    def _get_output_template(self, IMG, imgname, img_path):
        height, width, _ = IMG.shape
        afov_radians = (self.BEV_FOV / 2) * math.pi / 180
        focal_length_px = (max(width, height)/2) / math.tan(afov_radians)
        template = {
            'imgname': f'{imgname}.{self.image_format}',
            'imgpath': img_path,
            'img_height': height,
            'img_width': width,
            'cam_transl': [0., 0., 0.] ,
            'cam_rot': [0., 0., 180.],
            'fl': focal_length_px,
            'afov_horizontal': self.BEV_FOV,
        }
        return template

    def _load_single_human(
        self,
        op_data,
        vitpose_data,
        bev_data,
        op_human_idx,
        opbev_kpcost_best_match,
        opvitpose_kpcost_best_match,
        opvitpose_kpcost_matrix,
        img_height, 
        img_width,
    ):
        bev_human_idx = opbev_kpcost_best_match[op_human_idx]
        human_data, has_infant = self.process_bev(
            bev_human_idx, bev_data, (img_height, img_width))
        human_data['has_infant'] = has_infant
        
        # check if infant or no bev match was detected. If so, ignore image.
        if (human_data is None) or (bev_human_idx == -1): 
            return None

        # process OpenPose keypoints
        op_kpts = np.zeros((135, 3))
        if op_human_idx != -1:
            kpts = op_data[op_human_idx]
            # body + hands
            body = np.array(kpts['pose_keypoints_2d'] + \
                kpts['hand_left_keypoints_2d'] + kpts['hand_right_keypoints_2d']
            ).reshape(-1,3)
            # face 
            face = np.array(kpts['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
            contour = np.array(kpts['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[:17, :]
            # final openpose
            op_kpts = np.concatenate([body, face, contour], axis=0)

        # OpenPose and vit detection cost (if cost is too high, use Openpose) 
        vitpose_human_idx = opvitpose_kpcost_best_match[op_human_idx]
        
        # OpenPose and vit detection cost (if cost is too high, use Openpose)
        vitpose_kpts = np.zeros_like(op_kpts) 
        if vitpose_human_idx != -1:
            kpts = vitpose_data[vitpose_human_idx]
            # body + hands
            body = np.array(kpts['pose_keypoints_2d'] + \
                kpts['hand_left_keypoints_2d'] + kpts['hand_right_keypoints_2d']
            ).reshape(-1,3)
            # face 
            face = np.array(kpts['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
            contour = np.array(kpts['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[:17, :]
            # final openpose
            vitpose_kpts = np.concatenate([body, face, contour], axis=0)

        # # add keypoints vitposeplus
        # vitposeplus_kpts = np.zeros_like(op_kpts)
        # if vitposeplus_human_idx != -1:
        #     vitposeplus_kpts_orig = vitposeplus_data[vitposeplus_human_idx]['keypoints']
        #     main_body_idxs = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
        #     vitposeplus_kpts[main_body_idxs] = vitposeplus_kpts_orig[:17] # main body keypoints
        #     vitposeplus_kpts[19:25] = vitposeplus_kpts_orig[17:23] # foot keypoints
        #     vitposeplus_kpts[25:46] = vitposeplus_kpts_orig[-42:-21] # left hand keypoints
        #     vitposeplus_kpts[46:67] = vitposeplus_kpts_orig[-21:] # right hand keypoints
        #     #vitposeplus_kpts[67:135] = vitposeplus_kpts_orig[23:-42] # face keypoints
        #     face_countour = vitposeplus_kpts_orig[23:-42] 
        #     face = np.array(face_countour)[17: 17 + 51, :]
        #     contour = np.array(face_countour)[:17, :]
        #     vitposeplus_kpts[67:135] = np.concatenate([face, contour], axis=0) 

        # add idxs, bev data and keypoints to template
        human_data['openpose_human_idx'] = op_human_idx
        human_data['bev_human_idx'] = bev_human_idx
        human_data['vitpose_human_idx'] = vitpose_human_idx
        human_data['vitposeplus_human_idx'] = vitpose_human_idx
        human_data['vitpose'] = vitpose_kpts
        human_data['openpose'] = op_kpts
        human_data['vitposeplus'] = vitpose_kpts
    
        for k, v in human_data.items():

            if k in [
                'bev_smplx_global_orient', 'bev_smplx_body_pose', 'bev_smplx_transl', 
                'bev_smplx_keypoints', 'bev_smplx_vertices'
            ]:
                v = v[0]

            human_data[k] = np.array(v).copy()

        return human_data

    def load_single_image(self, imgname):

        img_path, IMG, bev_data, op_data, vitpose_data = self.read_data(imgname)        
        image_data = self._get_output_template(IMG, imgname, img_path)

        ################ Find all overlapping bounding boxes and process these people #################
        op_bbox = self.bbox_from_openpose(op_data, kp_key='pose_keypoints_2d')
        all_person_ids = []
        for bb1_idx in range(op_bbox.shape[0]):
            for bb2_idx in range(bb1_idx + 1, op_bbox.shape[0]):
                bb1, bb2 = op_bbox[bb1_idx], op_bbox[bb2_idx]
                bb12_iou = iou(bb1, bb2)
                if bb12_iou > 0:
                    all_person_ids.append([bb1_idx, bb2_idx])

        # cost matric to solve correspondance between openpose and bev and vitpose
        opbev_kpcost_matrix, opbev_kpcost_best_match = \
            self._get_opbev_cost(op_data, bev_data, IMG, self.unique_keypoint_match)
        opvitpose_kpcost_matrix, opvitpose_kpcost_best_match = \
            self._get_opvitpose_cost(op_data, vitpose_data, IMG, self.unique_keypoint_match)

        ################ load the two humans in contact for each pair #################
        all_image_data = []
        for pidx, person_ids in enumerate(all_person_ids):

            image_data['contact_index'] = pidx

            h0 = self._load_single_human(
                    op_data, vitpose_data, bev_data,
                    person_ids[0],
                    opbev_kpcost_best_match,
                    opvitpose_kpcost_best_match,
                    opvitpose_kpcost_matrix,
                    IMG.shape[0], IMG.shape[1],
            )

            h1 = self._load_single_human(
                    op_data, vitpose_data, bev_data,
                    person_ids[1],
                    opbev_kpcost_best_match,
                    opvitpose_kpcost_best_match,
                    opvitpose_kpcost_matrix,
                    IMG.shape[0], IMG.shape[1],
            )

            if h0 is None or h1 is None:
                return None

            concatenated_dict = {}
            for key in h0.keys():
                concatenated_dict[key] = np.stack((h0[key], h1[key]), axis=0)
            
            image_data.update(concatenated_dict)

            all_image_data.append(image_data.copy())

        return all_image_data


    def load(self):

        guru.info(f'Processing data from {self.data_folder}')

        data = []
        for imgname in os.listdir(self.image_folder):

            # ignore images that were not selected
            if self.image_name_select != '':
                if self.image_name_select not in imgname:
                    continue   

            data += self.load_single_image('.'.join(imgname.split('.')[:-1]))
            
        return data