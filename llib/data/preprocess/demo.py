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
        if self.has_gt_contact_annotation:
            self.imar_vision_datasets_tools_folder =  imar_vision_datasets_tools_folder
            annotation_fn = osp.join(
                self.data_folder, 'interaction_contact_signature.json'
            )
            if os.path.exists(annotation_fn):
                self.annotation = json.load(open(annotation_fn, 'r'))


            contact_regions_fn = osp.join(
                self.imar_vision_datasets_tools_folder, 'info/contact_regions.json'
            )
            contact_regions = json.load(open(contact_regions_fn, 'r'))
            self.rid_to_smplx_fids = contact_regions['rid_to_smplx_fids']

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
        image_data,
        human_id, 
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
        human_body_model_params = self.process_bev(
            bev_human_idx, bev_data, (img_height, img_width))

        # check if infant of no bev matach was detected. If so, ignore image.
        if (human_body_model_params is None) or (bev_human_idx == -1): 
            return None

        # add bev parameters
        for k, v in human_body_model_params.items(): 
            image_data[f'{k}_h{human_id}'] = v

        # add openpose keypoints
        op_human_kpts = op_data[op_human_idx]
        image_data[f'op_keypoints_h{human_id}'] = np.array(
            op_human_kpts['pose_keypoints_2d']).reshape(-1,3)

        # OpenPose and vit detection cost (if cost is too high, use Openpose) 
        vitpose_human_idx = opvitpose_kpcost_best_match[op_human_idx]
        vitpose_human_kpts = compare_and_select_openpose_vitpose(
            vitpose_data, op_human_kpts, opvitpose_kpcost_matrix,
            op_human_idx, vitpose_human_idx, KEYPOINT_COST_TRHESHOLD)
        image_data[f'vitpose_keypoints_h{human_id}'] = np.array(
            vitpose_human_kpts['pose_keypoints_2d']).reshape(-1,3)

        return image_data

    def load_image(self, imgname):

        img_path, IMG, bev_data, op_data, vitpose_data = self.read_data(imgname)
        image_data = self._get_output_template(IMG, imgname, img_path)

        ################ Match HHC annotation with OpenPose and ViT keypoints #################
        op_bbox = self.bbox_from_openpose(op_data, kp_key='pose_keypoints_2d') 

        # none of this available at test time without contact map
        image_data['contact_index'] = 0
        image_data['hhc_contacts_human_ids'] = [0, 1]
        image_data['hhc_contacts_region_ids'] = 0
        image_data['contact_map'] = self.contact_zeros

        person_ids = [0, 1] # just use the first two detections of openpose
        cop, cbev, cvit = len(op_bbox), bev_data['joints'].shape[0], len(vitpose_data)
        if cop != 2 or cbev != 2 or cvit != 2:
            print('Num bbox detected: ', cop, cbev, cvit)
            return None

        # cost matric to solve correspondance between openpose and bev and vitpose
        opbev_kpcost_matrix, opbev_kpcost_best_match = \
            self._get_opbev_cost(op_data, bev_data, IMG, self.unique_keypoint_match)
        opvitpose_kpcost_matrix, opvitpose_kpcost_best_match = \
            self._get_opvitpose_cost(op_data, vitpose_data, IMG, self.unique_keypoint_match)

        # Select the two bboxes with highest confidence
        ################ load the two humans in contact #################
        for human_id in person_ids:
            bbox = op_bbox[human_id] 
            image_data[f'bbox_h{human_id}'] = bbox
            # this is redundant, but keep in case bounding boxes are available
            #op_human_idx = ciop_iou_best_match[bbox_id]
            op_human_idx = human_id

            self._load_single_human(
                image_data, human_id, 
                op_data, vitpose_data, bev_data,
                op_human_idx,
                opbev_kpcost_best_match,
                opvitpose_kpcost_best_match,
                opvitpose_kpcost_matrix,
                IMG.shape[0], IMG.shape[1],
            )

        return [image_data]


    def load_from_cmap(self, imgname, annotation):
        
        img_path, IMG, bev_data, op_data, vitpose_data = self.read_data(imgname)
        image_data_template = self._get_output_template(IMG, imgname, img_path)

        ################ Match HHC annotation with OpenPose and ViT keypoints #################
        op_bbox = self.bbox_from_openpose(op_data, kp_key='pose_keypoints_2d') 
        ci_bbox = np.array(annotation['bbxes'])
        ciop_iou_matrix, ciop_iou_best_match = iou_matrix(ci_bbox, op_bbox)

        opbev_kpcost_matrix, opbev_kpcost_best_match = \
            self._get_opbev_cost(op_data, bev_data, IMG, False)
        opvitpose_kpcost_matrix, opvitpose_kpcost_best_match = \
            self._get_opvitpose_cost(op_data, vitpose_data, IMG, False)

        image_contact_data = []

        # load contact annotations
        for case_ci_idx, case_ci in enumerate(annotation['ci_sign']):
            IGNORE_PAIR = False
            image_data = image_data_template.copy()

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
                image_data[f'bbox_h{human_id}'] = bbox
                op_human_idx = ciop_iou_best_match[bbox_id]

                self._load_single_human(
                    image_data, human_id, 
                    op_data, vitpose_data, bev_data,
                    op_human_idx,
                    opbev_kpcost_best_match,
                    opvitpose_kpcost_best_match,
                    opvitpose_kpcost_matrix,
                    IMG.shape[0], IMG.shape[1],
                )

            image_contact_data.append(image_data)

        return image_contact_data


    def load(self):

        guru.info(f'Processing data from {self.data_folder}')

        count_failed = 0

        data = []
        # iterate though dataset / images
        if self.has_gt_contact_annotation:
            for imgname, anno in tqdm(self.annotation.items()):
                try:
                    data += self.load_from_cmap(imgname, anno)
                except:
                    print(f'Failed loading data for image {imgname}')
        else:
            for imgname in os.listdir(self.image_folder):
                if self.image_name_select != '':
                    if self.image_name_select not in imgname:
                        continue   
                try:
                    data += self.load_image('.'.join(imgname.split('.')[:-1]))
                except Exception as e:
                    print(e)
                    # is exception i skeyboard interrupt, exit program
                    if isinstance(e, KeyboardInterrupt):
                        sys.exit()
                    count_failed += 1
                    print(f'Failed loading data for image {imgname}')

        print(f'Loaded dataset: sucess {len(data)}, failed {count_failed}')

        return data
    
    def get_single_item(self, index):

        item = self.data[index]

        # crop image using both bounding boxes
        h1_bbox, h2_bbox = item[f'bbox_h0'], item[f'bbox_h1']
        bbox = self.join_bbox(h1_bbox, h2_bbox) if h1_bbox is not None else None
        # cast bbox to int
        bbox = np.array(bbox).astype(int)
        h1_bbox = np.array(h1_bbox).astype(int)
        h2_bbox = np.array(h2_bbox).astype(int)
        

        # Load image and resize directly before cropping, because of speed
        gen_target = {
            'images': [0.0],
            'imgpath': item['imgpath'],
            'bbox_h0': h1_bbox,
            'bbox_h1': h2_bbox,
            'bbox_h0h1': bbox,
            'imgname_fn': item['imgname'],
            'imgname_fn_out': os.path.splitext(item['imgname'])[0],
            'img_height': item['img_height'],
            'img_width': item['img_height'],
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


        h0id = 0 if item['transl_h0'][0] <= item['transl_h1'][0] else 1
        h1id = 1-h0id
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
            'keypoints_h0': item[f'vitpose_keypoints_h{h0id}'], # add vitpose as 'keypoints', since they're used
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
        
        

