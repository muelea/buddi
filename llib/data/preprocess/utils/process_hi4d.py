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
import argparse
from tqdm import tqdm
from llib.utils.image.bbox import iou_matrix
from llib.utils.keypoints.matching import keypoint_cost_matrix
from llib.defaults.body_model.main import conf as body_model_conf
from llib.cameras.perspective import PerspectiveCamera
from llib.bodymodels.utils import smpl_to_openpose
from loguru import logger as guru
from llib.data.preprocess.utils.shape_converter import ShapeConverter
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

KEYPOINT_COST_TRHESHOLD = 0.008

import torch
import torch.nn as nn
import glob

'''
Script to preprocess the original Hi4D dataset after running SMPL to SMPL-X conversion.
This script will create a processed.pkl file in the processed data folder that is used 
in the dataloader for the generatative model and that can be published with the dataset.
'''

def check_processing_status(orig_path='datasets/original/Hi4D', processed_path='datasets/processed/Hi4D'):
    print('pair images bev vit+ vit op smpl smplx')
    for pp in sorted(os.listdir(f'{orig_path}')):
        if not 'pair' in pp:
            continue
        images = len(glob.glob(f'{orig_path}/{pp}/**/images/**/*'))
        bev = len(glob.glob(f'{processed_path}/{pp}/**/bev/**/*'))
        vitplus = len(glob.glob(f'{processed_path}/{pp}/**/keypoints/vitposeplus/**/*'))
        vit = len(glob.glob(f'{processed_path}/{pp}/**/keypoints/vitpose/**/*'))
        op = len(glob.glob(f'{processed_path}/{pp}/**/openpose/keypoints/**/*'))
        smpl = len(glob.glob(f'{processed_path}/{pp}/**/smpl/*'))
        smplx = len(glob.glob(f'{processed_path}/{pp}/**/smplx/*'))
        print(pp, images, int((bev-32)/2), vitplus, vit, op, smpl, int(smplx/2))


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


class Hi4D():
    
    BEV_FOV = 60

    def __init__(
        self,
        original_data_folder,
        processed_data_folder,
        image_folder,
        bev_folder='bev',
        openpose_folder='openpose',
        body_model_type='smplx',
        vitpose_folder='vitpose',
        vitposeplus_folder='vitposeplus',
        correspondence_fn='correspondence.pkl',
        single_camera=False,
        single_camera_name='4'
    ):  

        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder

        self.body_model_type = body_model_type
        # placeholders for pair/action and camera
        self.image_folder = self.original_data_folder + '{}/{}' + image_folder + '{}'
        self.openpose_folder = self.processed_data_folder + '{}/{}' +  openpose_folder + '{}'
        self.bev_folder = self.processed_data_folder + '{}/{}' +   bev_folder + '{}'
        self.vitpose_folder = self.processed_data_folder + '{}/{}' +   vitpose_folder + '{}'
        self.vitposeplus_folder = self.processed_data_folder + '{}/{}' +   vitposeplus_folder + '{}'

        # correspondence openpose / vitpose / bev
        self.correspondence_fn = osp.join(self.processed_data_folder, correspondence_fn)
        self.correspondence = pickle.load(open(self.correspondence_fn, 'rb'))

        # load a single camera only 
        self.single_camera = single_camera
        self.single_camera_name = single_camera_name

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

        self.smpl_models = {}
        for gender in ['male', 'female']:
            self.smpl_models[gender] = smplx.create(
                model_path='essentials/body_models',
                model_type='smpl',
                gender=gender
            )
        self.smplx_model = {
            'neutral': smplx.create(
                model_path='essentials/body_models',
                model_type='smplx',
                gender='neutral'
            )
        }
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
                # this is bad because there is no SMIL-X model. We just do it for consistency.
                # But in this dataset, BEV shouldn\t detect infants anyway.
                smplx_betas_scale = self.shape_converter_smil.forward(torch.from_numpy(smpl_betas).unsqueeze(0))
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

    
    def load_smpl(self, smpl_path, genders):
        data = dict(np.load(smpl_path, allow_pickle=True))
        smpl_out_path = smpl_path.replace(
            self.original_data_folder, self.processed_data_folder)

        os.makedirs(osp.dirname(smpl_out_path), exist_ok=True)
        # write vertices as meshes (obj) per person
        for idx in [0,1]:
            verts = data['verts'][idx]
            smpl_out_path_idx = smpl_out_path.replace('.npz', f'_{idx}.obj')

            # comment in the next four lines to save obj file for conversion from SMPL to SMPl-X
            #if not osp.exists(smpl_out_path_idx):
            #    print('saving')
            #    mesh = trimesh.Trimesh(vertices=verts, faces=self.shape_converter_smpla.inbm.faces)
            #    _ = mesh.export(smpl_out_path_idx)

            # smpl verts from params (needs gender specific smpl model)
            #bm_male = smplx.create(model_path='essentials/body_models/', model_type='smpl', gender='male') 
            #body = bm_male(
            #    global_orient=torch.from_numpy(data['global_orient'][[idx]]), 
            #    body_pose=torch.from_numpy(data['body_pose'][[idx]]), 
            #    betas=torch.from_numpy(data['betas'][[idx]]), 
            #    transl=torch.from_numpy(data['transl'][[idx]])
            #)
            #smpl_out_path_idx = smpl_out_path.replace('.npz', f'_{idx}_smpl_neutral.obj')
            #mesh1 = trimesh.Trimesh(vertices=body.vertices.detach().cpu().numpy()[0], faces=self.shape_converter_smpla.inbm.faces)
            #_ = mesh1.export(smpl_out_path_idx)

        # convert to unit global orientation (we need 180 degree x-axis rotation)
        data['global_orient_unit'] = data['global_orient']
        data['transl_unit'] = data['transl']

        RR = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        for human_idx in [0,1]:
            betas =torch.from_numpy(data['betas'][[human_idx]]) 
            pelvis = self.smpl_models[genders[human_idx]](betas=betas).joints[:,0,:].detach().numpy()
            # apply rotation to smpl data
            global_orient = torch.from_numpy(np.array(data['global_orient'][human_idx]))
            global_orient = axis_angle_to_matrix(global_orient)
            global_orient = np.matmul(global_orient.transpose(1,0), RR.transpose(1 ,0)).transpose(1,0)
            global_orient = matrix_to_axis_angle(global_orient).unsqueeze(0)
            transl = np.array(data['transl'][human_idx])[None]
            transl = np.matmul(transl + pelvis, np.transpose(RR)) - pelvis
            data['global_orient_unit'][human_idx] = global_orient.numpy()
            data['transl_unit'][human_idx] = transl
            #body = self.smpl_models[genders[human_idx]](
            #    betas=betas.float(), 
            #    global_orient=global_orient.float(), 
            #    body_pose=torch.from_numpy(data['body_pose'][[human_idx]]).float(), 
            #    transl=torch.from_numpy(transl).float()
            #)
            #mesh = trimesh.Trimesh(vertices=body.vertices.detach().cpu().numpy()[0], faces=self.shape_converter_smpla.inbm.faces)
            #smpl_out_path_idx = f'outdebug/mesh_uni_glob_{human_idx}.obj'
            #_ = mesh.export(smpl_out_path_idx)

        #for human_idx in [0,1]:
        #    body = self.smpl_models[genders[human_idx]](
        #        betas=torch.from_numpy(data['betas'][[human_idx]]).float(), 
        #        global_orient=torch.from_numpy(data['global_orient'][[human_idx]]).float(), 
        #        body_pose=torch.from_numpy(data['body_pose'][[human_idx]]).float(), 
        #        transl=torch.from_numpy(data['transl'][[human_idx]]).float()
        #    )
        #    mesh = trimesh.Trimesh(vertices=body.vertices.detach().cpu().numpy()[0], faces=self.shape_converter_smpla.inbm.faces)
        #    smpl_out_path_idx = f'outdebug/mesh_uni_glob_{human_idx}_orig.obj'
        #    _ = mesh.export(smpl_out_path_idx) 

        return data

    def smplx_stack(self, x0, x1, to_aa):
        if to_aa:
            x0 = matrix_to_axis_angle(x0)
            x1 = matrix_to_axis_angle(x1)
        return np.concatenate([x0, x1], axis=0)

    def load_smplx(self, smplx_folder, contact_idx):

            smplx_subj0_path = osp.join(smplx_folder, f'{contact_idx:06d}_0.pkl')
            smplx_data0 = pickle.load(open(smplx_subj0_path, 'rb'))
            smplx_subj1_path = osp.join(smplx_folder, f'{contact_idx:06d}_1.pkl')
            smplx_data1 = pickle.load(open(smplx_subj1_path, 'rb'))

            data = {}
            for k in ['betas', 'global_orient', 'body_pose', 'transl']:
                data[f'smplx_{k}'] = self.smplx_stack(
                    smplx_data0[k].detach().cpu(), smplx_data1[k].detach().cpu(), 
                    to_aa=(k in ['body_pose', 'global_orient'])
                )

            # convert to unit global orientation (we need 180 degree x-axis rotation)
            data['smplx_global_orient_unit'] = data['smplx_global_orient'].squeeze(1)
            data['smplx_transl_unit'] = data['smplx_transl']
 
            RR = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            for human_idx in [0,1]:
                betas =torch.from_numpy(data['smplx_betas'][[human_idx]]) 
                pelvis = self.smplx_model['neutral'](betas=betas[:,:10]).joints[:,0,:].detach().numpy()
                # apply rotation to smpl data
                global_orient = torch.from_numpy(np.array(data['smplx_global_orient'].squeeze(1)[human_idx].reshape(1,3)))[0]
                global_orient = axis_angle_to_matrix(global_orient)
                global_orient = np.matmul(global_orient.transpose(1,0), RR.transpose(1 ,0)).transpose(1,0)
                global_orient = matrix_to_axis_angle(global_orient)
                transl = np.array(data['smplx_transl'][human_idx])[None]
                transl = np.matmul(transl + pelvis, np.transpose(RR)) - pelvis
                data['smplx_global_orient_unit'][human_idx] = global_orient.numpy()
                data['smplx_transl_unit'][human_idx] = transl[0]

            return data

    def process_action(self, pair, action):
        action_folder = osp.join(self.original_data_folder, pair, action)
        action_folder_processed = osp.join(self.processed_data_folder, pair, action)
        smplx_folder = osp.join(action_folder_processed, 'smplx')
            
        meta = dict(np.load(osp.join(action_folder, 'meta.npz'), allow_pickle=True))
        contact_ids = meta['contact_ids']
        output = {
            'action_path': [],
            'betas': [],
            'global_orient': [],
            'body_pose': [],
            'transl': [],
            'smplx_betas': [],
            'smplx_global_orient': [],
            'smplx_body_pose': [],
            'smplx_global_orient_unit': [],
            'smplx_transl': [],
            'smplx_transl_unit': [],
            'image_data': []
        }
        for cid in contact_ids:
            smpl_path = osp.join(action_folder, f'smpl/{cid:06d}.npz') 
            # load smpl data
            try:
                smpl_data = self.load_smpl(smpl_path, meta['genders'])
                output['action_path'].append(action_folder) 
                for k, v in smpl_data.items():
                    if k in output.keys():
                        output[k].append(v)
            except:
                print("Error loading smpl data for ", smpl_path)
                continue
            
            # load smplx data
            try:
                smplx_data = self.load_smplx(smplx_folder, cid)
                for k, v in smplx_data.items():
                    if k in output.keys():
                        output[k].append(v)
            except:
                print("Error loading smplx data for ", smplx_folder)
                continue

            # match the flickr bbox with the openpose bbox
            cid_image_data = {}
            for cam in os.listdir(osp.join(action_folder, 'images')):
                if cam.startswith('.'):
                    continue
                if self.single_camera and not cam == self.single_camera_name:
                    continue
                cid_image_data[cam] = {}
                camera_image_folder = osp.join(action_folder, 'images', cam)
                #for image in os.listdir(camera_image_folder):
                if True:
                    image = f'{cid:06d}.jpg'
                    image_raw = image.split('.')[0]
                    cid_image_data[cam][image_raw] = {}
                    bev_path = self.bev_folder.format(f'/{pair}/{action}', '', f'/{cam}/{image_raw}__2_0.08.npz')
                    openpose_path = self.openpose_folder.format(f'/{pair}/{action}', '', f'/{cam}/{image_raw}.json')
                    vitpose_path = self.vitpose_folder.format(f'/{pair}/{action}', '', f'/{cam}/{image_raw}_keypoints.json') 
                    vitposeplus_path = self.vitposeplus_folder.format(f'/{pair}/{action}', '', f'/{cam}/{image_raw}.pkl')
                    image_path = self.image_folder.format(f'/{pair}/{action}', '', f'/{cam}/{image}') 

                    op_data = json.load(open(openpose_path, 'r'))['people']
                    vitpose_data = json.load(open(vitpose_path, 'r'))['people']
                    vitposeplus_data = pickle.load(open(vitposeplus_path, 'rb'))
                    # load bev
                    if os.path.exists(bev_path):
                        bev_data = np.load(bev_path, allow_pickle=True)['results'][()]
                    else:
                        bev_data = None

                    for human_idx in [0, 1]:
                        temp = {}
                        try:
                            op_human_idx = self.correspondence[pair][action][cam][image_raw]['openpose']['best_match'][human_idx]
                        except:
                            op_human_idx = -1
                        try:
                            bev_human_idx = self.correspondence[pair][action][cam][image_raw]['bev']['best_match'][human_idx] 
                        except:
                            bev_human_idx = -1
                        try:
                            vitpose_human_idx = self.correspondence[pair][action][cam][image_raw]['vitpose']['best_match'][human_idx] 
                        except:
                            vitpose_human_idx = -1
                        try:
                            vitposeplus_human_idx = self.correspondence[pair][action][cam][image_raw]['vitposeplus']['best_match'][human_idx] 
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

                        # process ViTPose keypoints
                        vitpose_kpts = np.zeros((135, 3))
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

                        #IMG = cv2.imread(image_path)
                        #plot_over_image(IMG, 
                        #points_2d=[op_kpts, vitposeplus_kpts, bev_params['bev_smpl_joints_2d']], 
                        #path_to_write=f'outdebug/hi4d/test_{image_raw}_{human_idx}.png')

                        # temp to arrays 
                        for k, v in temp.items():
                            temp[k] = np.array(v)

                        cid_image_data[cam][image_raw][human_idx] = temp
            output['image_data'].append(cid_image_data)

        for k, v in output.items():
            # stack to arrays
            if k != 'image_data':
                output[k] = np.stack(v, axis=0)
        
        return output

    def process(self, output_fn=f'processed.pkl'):

        processed_data_path = osp.join(
            self.processed_data_folder, output_fn
        )

        guru.info(f'Processing data from {self.original_data_folder}')

        pairs = os.listdir(self.original_data_folder)
        output = {}
        for pair in tqdm(pairs):
            if pair[:4]  != 'pair' or pair.endswith('.gz') or pair.startswith('.'):
                continue
            
            output[pair] = {}
            # get pair folder
            pair_folder = osp.join(self.original_data_folder, pair)
            actions = os.listdir(pair_folder)
            for action in tqdm(actions):
                if action.startswith('.'):
                    continue
                #action_folder = osp.join(pair_folder, action)
                action_data = self.process_action(pair, action)
                output[pair][action] = action_data
        
        # save data to processed data folder
        with open(processed_data_path, 'wb') as f:
            pickle.dump(output, f)
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--single_camera', action='store_true', help='Process a single camera only')
    args = parser.parse_args()
   
    original_data_folder = 'datasets/original/Hi4D'
    processed_data_folder = 'datasets/processed/Hi4D'

    #check_processing_status()

    if args.single_camera:
        single_camera = True
        single_camera_name = '4'
    else:
        single_camera = False
        single_camera_name = ''

    output_fn = 'processed_single_camera.pkl' if single_camera else 'processed.pkl'
    correspondence_fn = 'correspondence_single_camera.pkl' if single_camera else 'correspondence.pkl'

    hi4d_data = Hi4D(
        original_data_folder,
        processed_data_folder,
        image_folder='images',
        bev_folder='bev',
        openpose_folder='openpose/keypoints',
        body_model_type='smplx',
        vitpose_folder='keypoints/vitpose',
        vitposeplus_folder='keypoints/vitposeplus',
        correspondence_fn=correspondence_fn,
        single_camera=single_camera,
        single_camera_name=single_camera_name
    )
   
    hi4d_data.process(output_fn)