import os.path as osp 
import json 
import torch
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
from llib.utils.image.bbox import iou_matrix
from loguru import logger as guru
from llib.data.preprocess.utils.shape_converter import ShapeConverter
from pytorch3d.transforms import matrix_to_axis_angle
from llib.utils.keypoints.matching import keypoint_cost_matrix

KEYPOINT_COST_TRHESHOLD = 0.008
IMGNORM = 900 # CHI3D images are 900x900 resolution
import torch
import torch.nn as nn


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

class CHI3D():
    
    BEV_FOV = 60

    def __init__(
        self,
        original_data_folder,
        processed_data_folder,
        imar_vision_datasets_tools_folder,
        image_folder,
        joints3d_folder='joints3d_25',
        bev_folder='bev',
        openpose_folder='openpose',
        split='train',
        body_model_type='smplx',
        vitpose_folder='vitpose',
        vitposeplus_folder='vitposeplus',
        correspondence_fn='correspondence.pkl',
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder
        self.imar_vision_datasets_tools_folder =  imar_vision_datasets_tools_folder
        self.split = split
        self.split_folder = 'test' if split == 'test' else 'train'  

        self.joints3d_folder = joints3d_folder
        self.body_model_type = body_model_type
        self.image_folder = image_folder
        self.openpose_folder = openpose_folder
        self.bev_folder = bev_folder
        self.vitpose_folder = vitpose_folder
        self.vitposeplus_folder = vitposeplus_folder

        # correspondence openpose / vitpose / bev
        self.correspondence_fn = osp.join(self.processed_data_folder, self.split_folder, correspondence_fn)
        self.correspondence = pickle.load(open(self.correspondence_fn, 'rb'))

        # contact maps
        contact_regions_fn = osp.join(
            self.imar_vision_datasets_tools_folder, 'info/contact_regions.json')
        contact_regions = json.load(open(contact_regions_fn, 'r'))
        self.rid_to_smplx_fids = contact_regions['rid_to_smplx_fids']

        # Get SMPL-X pose, if available
        self.global_orient = torch.zeros(3, dtype=torch.float32)
        self.body_pose = torch.zeros(63, dtype=torch.float32)
        self.betas = torch.zeros(10, dtype=torch.float32)
        self.transl = torch.zeros(3, dtype=torch.float32)

        # keypoints 
        self.keypoints = torch.zeros((24, 3), dtype=torch.float32)

        # convert smpl betas to smpl-x betas 
        self.shape_converter_smpla = ShapeConverter(inbm_type='smpla', outbm_type='smplxa')
        self.shape_converter_smil = ShapeConverter(inbm_type='smil', outbm_type='smplxa')

        # create body model to get bev root translation from pose params
        self.body_model = self.shape_converter_smpla.outbm

        # output format 
        self.output = {}
        #'image_name':[],
        #'flickr_person_id':[],
        
        self.bev_params_template = {
            'bev_smpl_global_orient': np.zeros(3),
            'bev_smpl_body_pose': np.zeros(69),
            'bev_smpl_betas': np.zeros(10),
            'bev_smpl_scale': 0,
            'bev_smplx_betas': np.zeros(10),
            'bev_smplx_scale': 0,
            'bev_cam_trans': np.zeros(3),
            'bev_smpl_joints': np.zeros((71,3)),
            'bev_smpl_vertices': np.zeros((6890,3)),
            'bev_smpl_joints_2d': np.zeros((71,2)),
        }

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

    def process_bev(self, bev_human_idx, bev_data):

            smpl_betas_scale = bev_data['smpl_betas'][bev_human_idx]
            smpl_betas = smpl_betas_scale[:10]
            smpl_scale = smpl_betas_scale[-1]
            smpl_body_pose = bev_data['smpl_thetas'][bev_human_idx][3:]
            smpl_global_orient = bev_data['smpl_thetas'][bev_human_idx][:3]

            if smpl_scale > 0.8:
                # this is bad, because there is no SMIL-X Model. We just do it for consistency.
                # In this dataset BEV shouldn't detect infants anyway.
                smplx_betas = self.shape_converter_smil.forward(torch.from_numpy(smpl_betas).unsqueeze(0))
                smplx_betas = smplx_betas_scale[0,:10].numpy()
                smplx_scale = smplx_betas_scale[0,10].numpy()
            else:
                smplx_betas_scale = self.shape_converter_smpla.forward(torch.from_numpy(smpl_betas_scale).unsqueeze(0))
                smplx_betas = smplx_betas_scale[0,:10].numpy()
                smplx_scale = smplx_betas_scale[0,10].numpy()
            
            cam_trans = bev_data['cam_trans'][bev_human_idx]
            smpl_joints = bev_data['joints'][bev_human_idx]
            smpl_vertices = bev_data['verts'][bev_human_idx]
            smpl_joints_2d = bev_data['pj2d_org'][bev_human_idx]

            params = {
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

            return params
    
    
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
            bev_path = osp.join(processed_subj_folder, self.bev_folder, f'{imgname}__2_0.08.npz')
            openpose_path = osp.join(processed_subj_folder,self.openpose_folder, f'{imgname}.json')
            vitpose_path = osp.join(processed_subj_folder,self.vitpose_folder, f'{imgname}_keypoints.json')
            vitposeplus_path = osp.join(processed_subj_folder,self.vitposeplus_folder, f'{imgname}.pkl')
            
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
            
            # load bev
            if os.path.exists(bev_path):
                bev_data = np.load(bev_path, allow_pickle=True)['results'][()]
            else:
                bev_data = None
        
            # load keypoints
            op_data = json.load(open(openpose_path, 'r'))['people']
            vitpose_data = json.load(open(vitpose_path, 'r'))['people']
            vitposeplus_data = pickle.load(open(vitposeplus_path, 'rb'))

            self.output[subject][imgname] = {}


            for bbox_id, bbox in enumerate(j2d_camera):

                #if bbox_id not in unique_person_ids:
                #    continue

                temp = {}

                # match the flickr bbox with the openpose bbox
                try:
                    op_human_idx = self.correspondence[subject][imgname]['openpose']['best_match'][bbox_id]
                except:
                    op_human_idx = -1
                
                try:
                    bev_human_idx = self.correspondence[subject][imgname]['bev']['best_match'][bbox_id] 
                except:
                    bev_human_idx = -1

                try:
                    vitpose_human_idx = self.correspondence[subject][imgname]['vitpose']['best_match'][bbox_id] 
                except:
                    vitpose_human_idx = -1

                try:
                    vitposeplus_human_idx = self.correspondence[subject][imgname]['vitposeplus']['best_match'][bbox_id] 
                except:
                    vitposeplus_human_idx   = -1    

                # process BEV data
                bev_params = self.bev_params_template.copy()
                if bev_human_idx != -1 and bev_data is not None:
                    bev_params = self.process_bev(bev_human_idx, bev_data)

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

                # add keypoints vitposeplus
                vitposeplus_kpts = np.zeros_like(op_kpts)
                if vitposeplus_human_idx != -1:
                    vitposeplus_kpts_orig = vitposeplus_data[vitposeplus_human_idx]['keypoints']
                    main_body_idxs = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
                    vitposeplus_kpts[main_body_idxs] = vitposeplus_kpts_orig[:17] # main body keypoints
                    vitposeplus_kpts[19:25] = vitposeplus_kpts_orig[17:23] # foot keypoints
                    vitposeplus_kpts[25:46] = vitposeplus_kpts_orig[-42:-21] # left hand keypoints
                    vitposeplus_kpts[46:67] = vitposeplus_kpts_orig[-21:] # right hand keypoints
                    #vitposeplus_kpts[67:135] = vitposeplus_kpts_orig[23:-42] # face keypoints
                    face_countour = vitposeplus_kpts_orig[23:-42] 
                    face = np.array(face_countour)[17: 17 + 51, :]
                    contour = np.array(face_countour)[:17, :]
                    vitposeplus_kpts[67:135] = np.concatenate([face, contour], axis=0) 

                # add idxs, bev data and keypoints to template
                #temp['chi3d_bbox'] = bbox
                temp['openpose_human_idx'] = op_human_idx
                temp['bev_human_idx'] = bev_human_idx
                temp['vitpose_human_idx'] = vitpose_human_idx
                temp['vitposeplus_human_idx'] = vitposeplus_human_idx

                for k, v in bev_params.items(): 
                    temp[k] = np.array(v, dtype=np.float32)

                temp['openpose'] = op_kpts
                temp['vitpose'] = vitpose_kpts
                temp['vitposeplus'] = vitposeplus_kpts

                #IMG = cv2.imread(img_path)
                #plot_over_image(IMG, points_2d=[op_kpts, vitpose_kpts, vitposeplus_kpts, bbox, bev_params['bev_smpl_joints_2d']], path_to_write=f'outdebug/chi3d/test_{imgname}_{bbox_id}.png')

                # temp to arrays 
                for k, v in temp.items():
                    temp[k] = np.array(v)
                
                # add to full clean dataset
                self.output[subject][imgname][bbox_id] = temp
        


    def process(self):

        processed_data_path = osp.join(
            self.processed_data_folder, self.split_folder, f'images_contact_processed.pkl'
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
    imar_vision_datasets_tools_folder = 'essentials/imar_vision_datasets_tools'
    split = 'train'

    chi3d_data = CHI3D(
        original_data_folder,
        processed_data_folder,
        imar_vision_datasets_tools_folder,
        image_folder='images',
        bev_folder='bev',
        openpose_folder='images_contact_openpose/keypoints',
        split=split,
        body_model_type='smplx',
        vitpose_folder='images_contact_vitpose',
        vitposeplus_folder='images_contact_vitposeplus',
        correspondence_fn='images_contact_correspondence.pkl',
    )
   
    chi3d_data.process()
