import os.path as osp 
import json 
import torch
import numpy as np
import os
import cv2
import math
import pickle
from llib.cameras.perspective import PerspectiveCamera
from llib.data.preprocess.utils.shape_converter import ShapeConverter
from loguru import logger as guru
from pytorch3d.transforms import matrix_to_axis_angle

KEYPOINT_COST_TRHESHOLD = 0.008

import torch

class CHI3D():
    
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
        BODY_MODEL_FOLDER = 'essentials/body_models',
        load_single_camera=False,
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder
        self.imar_vision_datasets_tools_folder =  imar_vision_datasets_tools_folder
        self.split = split
        self.load_single_camera = load_single_camera
        
        # validation data/images must be loaded from training folder
        self.split_folder = 'test' if split == 'test' else 'train'
        if self.split_folder == 'train':
            trainval_fn = osp.join(self.processed_data_folder, 'train_val_split.npz')
            self.subjects = np.load(trainval_fn)[self.split]

        self.body_model_type = body_model_type
        self.image_folder = image_folder
        self.openpose_folder = openpose_folder
        self.bev_folder = bev_folder
        self.vitpose_folder = vitpose_folder
        self.pseudogt_folder = pseudogt_folder
        self.has_pseudogt = False if pseudogt_folder == '' else True

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

        # create body model to get bev root translation from pose params
        # convert smpl betas to smpl-x betas 
        shape_converter = ShapeConverter(inbm_type='smpla', outbm_type='smplxa')
        self.body_model = shape_converter.outbm

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

    def read_cam_params(self, cam_path):
        with open(cam_path) as f:
            cam_params = json.load(f)
            for key1 in cam_params:
                for key2 in cam_params[key1]:
                    cam_params[key1][key2] = np.array(cam_params[key1][key2]) 
        return cam_params

    def load_smpl_data(self, smpl_fn, frame_id=85):
        smplx_data = json.load(open(smpl_fn, 'r'))
        human_target = {}
        for human_idx in [0, 1]:
            betas = torch.from_numpy(np.array(smplx_data['betas'][human_idx][frame_id])).unsqueeze(0).float()
            betas_with_scale = torch.cat((betas, torch.zeros(1, 1)), dim=-1)
            # bring mesh into same position as BEV estimate 
            # get pelvis of person with markers
            pelvis = self.body_model(betas=betas_with_scale).joints[:, 0, :].detach().numpy()
            # create rotation matrix for 90 degree x axis rotation 
            RR = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            # apply rotation to smpl data
            global_orient = np.array(smplx_data['global_orient'][human_idx][frame_id])
            global_orient = np.matmul(global_orient.transpose(0, 2, 1), RR.transpose(1 ,0)).transpose(0,2,1)
            transl = np.array(smplx_data['transl'][human_idx][frame_id])[None]
            transl = np.matmul(transl + pelvis, np.transpose(RR)) - pelvis
            transl = torch.from_numpy(transl).float()
            # update human target dict with smpl data
            params = {
                f'global_orient_h{human_idx}': matrix_to_axis_angle(torch.from_numpy(global_orient)).float(),
                f'transl_h{human_idx}': transl,
                f'translx_h{human_idx}': transl[:,0].unsqueeze(1),
                f'transly_h{human_idx}': transl[:,1].unsqueeze(1),
                f'translz_h{human_idx}': transl[:,2].unsqueeze(1),
                f'body_pose_h{human_idx}': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['body_pose'][human_idx][frame_id]))).flatten().unsqueeze(0).float(), 
                f'betas_h{human_idx}': betas_with_scale,
                f'left_hand_pose_h{human_idx}': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['left_hand_pose'][human_idx][frame_id]))).flatten().unsqueeze(0).float()[:,:6],
                f'right_hand_pose_h{human_idx}': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['right_hand_pose'][human_idx][frame_id]))).flatten().unsqueeze(0).float()[:,:6], 
                f'jaw_pose_h{human_idx}': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['jaw_pose'][human_idx][frame_id]))).float(), 
                f'leye_pose_h{human_idx}': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['leye_pose'][human_idx][frame_id]))).float(), 
                f'reye_pose_h{human_idx}': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['reye_pose'][human_idx][frame_id]))).float(), 
                f'expression_h{human_idx}': torch.from_numpy(np.array(smplx_data['expression'][human_idx][frame_id])).unsqueeze(0).float(),
            }
            params_for_smpl = {}
            for key, val in params.items():
                params_for_smpl[key.replace(f'_h{human_idx}', '')] = val
            body = self.body_model(**params_for_smpl)
            # save mesh for debugging 
            #import trimesh
            #mm = trimesh.Trimesh(body.vertices.detach().to('cpu').numpy()[0],self.body_model.faces)
            #mm.export(f'meshes/m{human_idx}_63_transformed_[1, 0, 0], [0, 0, -1], [0, 1, 0].obj')

            params_ = params.copy()
            for k, v in params_.items():
                params[k] = v.detach().to('cpu').numpy()[0]
                if 'betas' in k:
                    params[f'pseudogt_{k}'] = v.detach().to('cpu').numpy()[0][:10]
                else:
                    params[f'pseudogt_{k}'] = v.detach().to('cpu').numpy()[0]

            params[f'vertices_h{human_idx}'] = body.vertices.detach().to('cpu').numpy()[0]
            params[f'joints_h{human_idx}'] = body.joints.detach().to('cpu').numpy()[0]
            params[f'bev_keypoints_h{human_idx}'] = np.zeros((127, 2))
            params[f'bev_orig_vertices_h{human_idx}'] = np.zeros((6890, 3))
            params[f'op_keypoints_h{human_idx}'] = np.zeros((25, 3))
            params[f'vitpose_keypoints_h{human_idx}'] = np.zeros((25, 3))


            human_target.update(params)
        return human_target

    def load_single_image(self, subject, action, annotation):

        orig_subject_folder = osp.join(self.original_data_folder, self.split_folder, subject)
        processed_subject_folder = osp.join(self.processed_data_folder, self.split_folder, subject)

        frame_id = annotation['fr_id']

        # get camera parameters 
        cameras = os.listdir(osp.join(orig_subject_folder, 'camera_parameters'))

        # load SMPL params
        smpl_path = f'{orig_subject_folder}/smplx/{action}.json'
        smpl_params = self.load_smpl_data(smpl_path, frame_id)

        ################ load the two humans in contact #################
        smpl_params[f'bbox_h0'] = np.array([0,0,0,0])
        smpl_params[f'bbox_h1'] = np.array([0,0,0,0])

        #if self.load_single_camera:
        #    cameras = ['58860488'] #cameras[:1]

        data = []
        for cam in cameras:
            cam_path = osp.join(orig_subject_folder, 'camera_parameters', cam, f'{action}.json')
            cam_params = self.read_cam_params(cam_path)
            cam_intrinsics = cam_params['intrinsics_wo_distortion']
            TT = torch.tensor(cam_params['extrinsics']['T']).to("cuda")
            RR = torch.tensor(cam_params['extrinsics']['R']).to("cuda")

            img_name = f'{action}_{frame_id:06d}_{cam}'
            img_path = osp.join(processed_subject_folder, self.image_folder, img_name+'.jpg')
            if not osp.exists(img_path):
                # use no image if frames were not extracted
                IMG = np.zeros((900, 900, 3)).astype(np.uint8)
            else:
                IMG = cv2.imread(img_path)
            
            ################ camera params #################
            height, width, _ = IMG.shape
            # camera translation was already applied to mesh, so we can set it to zero.
            cam_transl = [0., 0., 0.] 
            # camera rotation needs 180 degree rotation around z axis, because bev and
            # pytorch3d camera coordinate systems are different            
            cam_rot = [0., 0., 180.]

            afov_radians = (self.BEV_FOV / 2) * math.pi / 180
            focal_length_px = (max(width, height)/2) / math.tan(afov_radians)

            data_item = {
                'imgname': f'{img_name}.png',
                'imgpath': img_path,
                'img_height': height,
                'img_width': width,
                'cam_transl': cam_transl,
                'cam_rot': cam_rot,
                'fl': focal_length_px,
                'afov_horizontal': self.BEV_FOV,
                }

            data_item.update(**smpl_params)
            data.append(data_item)
    
        return data

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
            #for imgname, anno in tqdm(self.annotation.items()):
            for subject in self.subjects:
                annotation_fn =  osp.join(
                    self.original_data_folder, self.split_folder, subject, 'interaction_contact_signature.json'
                )
                annotation = json.load(open(annotation_fn, 'r'))

                for action, anno in annotation.items():
                    data += self.load_single_image(subject, action, anno) 

                    #try:
                    #    data += self.load_single_image(action, anno) 
                    #except Exception as e:                
                        # if exeption is keyboard interrupt end program
                        #if isinstance(e, KeyboardInterrupt):
                        #    raise e                    
                        #else:
                        #    print(f'Error loading {imgname}')
                        #    print(f'Exception: {e}')
                        #    continue

            # save data to processed data folder
            with open(processed_data_path, 'wb') as f:
                pickle.dump(data, f)

        if self.overfit:
            data = data[:self.overfit_num_samples]

        if self.load_single_camera:
            data = [x for x in data if '58860488' in x['imgpath']]

        return data
        