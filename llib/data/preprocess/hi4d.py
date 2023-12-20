import os.path as osp 
import json 
import torch
import numpy as np
import os
from tqdm import tqdm
import cv2
import smplx
import math
import pickle
import trimesh
from llib.cameras.perspective import PerspectiveCamera
from llib.data.preprocess.utils.shape_converter import ShapeConverter
from loguru import logger as guru
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix

KEYPOINT_COST_TRHESHOLD = 0.008

import torch

class HI4D():
    
    BEV_FOV = 60

    def __init__(
        self,
        original_data_folder,
        processed_data_folder,
        split='train',
        overfit=False,
        overfit_num_samples=12,
        load_single_camera=False,
        load_from_scratch_single_camera: bool = False,
        body_model_type='smplx',
        load_unit_glob_and_transl=False,
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder
        #self.imar_vision_datasets_tools_folder =  imar_vision_datasets_tools_folder
        self.split = split
        self.load_single_camera = load_single_camera
        self.load_unit_glob_and_transl = load_unit_glob_and_transl
        self.load_from_scratch_single_camera = load_from_scratch_single_camera
        self.body_model_type = body_model_type
        
        # validation data/images must be loaded from training folder
        self.split_folder = 'test' if split == 'test' else 'train'
        trainval_fn = osp.join(self.processed_data_folder, 'train_val_test_split.npz')
        self.subjects = np.load(trainval_fn)[self.split]

        # for overfitting experiments we use the first 12 samples
        self.overfit = overfit
        self.overfit_num_samples = overfit_num_samples

        if self.load_single_camera:
            processed_pkl_fn = osp.join(self.processed_data_folder, 'processed_single_camera.pkl')
        else:
            processed_pkl_fn = osp.join(self.processed_data_folder, 'processed.pkl')
        self.processed = pickle.load(open(processed_pkl_fn, 'rb'))

        shape_converter = ShapeConverter(inbm_type='smpla', outbm_type='smplxa')
        self.body_model = shape_converter.outbm

        self.body_models_hi4d = smplx.create(
            model_path='essentials/body_models',
            model_type='smplx',
            gender='neutral',
        )

        # Get SMPL-X pose, if available
        self.global_orient = torch.zeros(3, dtype=torch.float32)
        self.body_pose = torch.zeros(63, dtype=torch.float32)
        self.betas = torch.zeros(10, dtype=torch.float32)
        self.transl = torch.zeros(3, dtype=torch.float32)

        # keypoints 
        self.keypoints = torch.zeros((24, 3), dtype=torch.float32)

    def concatenate_dicts(self, x, y):
        concatenated_dict = {}
        for key in x.keys():
            try:
                concatenated_dict[key] = np.stack((x[key], y[key]), axis=0)
            except:
                import ipdb; ipdb.set_trace()
        return concatenated_dict

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
                body_pose = data['bev_smpl_body_pose'][:,3:]
                global_orient = data['bev_smpl_body_pose'][:,:3]
                betas = data['bev_smpl_betas']
                scale = data['bev_smpl_scale']

            # ignore infants, because SMPL-X doesn't support them (template is noisy)
            if np.any(scale > 0.8):
                return None 
            
            bev_cam_trans = torch.from_numpy(data['bev_cam_trans'])
            if not self.load_params_only:
                bev_camera = PerspectiveCamera(
                    rotation=torch.tensor([[0., 0., 180.]]),
                    translation=torch.tensor([[0., 0., 0.]]),
                    afov_horizontal=torch.tensor([self.BEV_FOV]),
                    image_size=torch.tensor([[width, height]]),
                    batch_size=1,
                    device='cpu'
                )
                # bev_vertices = data['bev_smpl_vertices']
                # bev_root_trans = data['bev_smpl_joints'][:,[45,46],:].mean(1)
                # bev_vertices_root_trans = bev_vertices - bev_root_trans[:,np.newaxis,:] \
                    # + bev_cam_trans.numpy()[:,np.newaxis,:]
                # data['bev_smpl_vertices_root_trans'] = bev_vertices_root_trans
            
            smplx_update = {
                'bev_smplx_global_orient': [],
                'bev_smplx_body_pose': [],
                'bev_smplx_transl': []
            }
            if not self.load_params_only:
                smplx_update.update({
                    'bev_smplx_keypoints': [],
                    'bev_smplx_vertices': [],
                })

            for idx in range(2):

                h_global_orient = torch.from_numpy(global_orient[[idx]])
                smplx_update['bev_smplx_global_orient'].append(h_global_orient)
                
                h_body_pose = torch.from_numpy(body_pose[[idx]])
                smplx_update['bev_smplx_body_pose'].append(h_body_pose)

                h_betas_scale = torch.from_numpy(
                    np.concatenate((betas[[idx]], scale[[idx]][None]), axis=1)
                )

                body = self.body_model(
                    global_orient=h_global_orient,
                    body_pose=h_body_pose,
                    betas=h_betas_scale,
                )

                root_trans = body.joints.detach()[:,0,:]
                transl = -root_trans.to('cpu') + bev_cam_trans[[idx]].to('cpu')
                smplx_update['bev_smplx_transl'].append(transl)

                if not self.load_params_only:
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

            return data

    def load_single_image(self, subject, action):

        data = self.processed['pair'+subject][action]

        IMG = np.zeros((1280, 940, 3)).astype(np.uint8)
        
        ################ camera params #################
        height, width, _ = IMG.shape
        # camera translation was already applied to mesh, so we can set it to zero.
        cam_transl = [0., 0., 0.] 
        # camera rotation needs 180 degree rotation around z axis, because bev and
        # pytorch3d camera coordinate systems are different            
        cam_rot = [0., 0., 180.]

        afov_radians = (self.BEV_FOV / 2) * math.pi / 180
        focal_length_px = (max(width, height)/2) / math.tan(afov_radians)

        data_out = []

        # Hi4D camera data
        action_folder = osp.join(self.original_data_folder, 'pair'+subject, action)
        camera_data = np.load(osp.join(action_folder, 'cameras/rgb_cameras.npz'))       
        def get_current_cam(camera_data, cam_idx):
            array_idx = np.where(camera_data['ids'] == int(cam_idx))
            out = {}
            for key in camera_data.keys():
                out[key] = camera_data[key][array_idx]
            return out

        # Hi4D meta data (npz)
        # we use the SMPL-X converted parameters from the original Hi4D dataset
        # so the gender is neutral instead of male/female
        # meta_data = np.load(osp.join(action_folder, 'meta.npz'))      


        for idx in np.arange(0, len(data['betas']), 5):
            data_item = {
                #'imgname': f'{idx:06d}.jpg',
                #'imgpath': os.path.join(data['action_path'][idx], f'images/4/{idx:06d}.jpg'),
                'img_height': height,
                'img_width': width,
                'cam_transl': cam_transl,
                'cam_rot': cam_rot,
                'fl': focal_length_px,
                'afov_horizontal': self.BEV_FOV,
                'betas_smpl': data['betas'][idx],
                'global_orient_smpl': data['global_orient'][idx],
                'body_pose_smpl': data['body_pose'][idx],
                'transl_smpl': data['transl'][idx],                
                'betas_smplx': data['smplx_betas'][idx][:,:10],
                'global_orient_smplx': data['smplx_global_orient_unit'][idx].reshape(2, -1),
                'body_pose_smplx': data['smplx_body_pose'][idx].reshape(2, -1),
                'transl_smplx': data['smplx_transl_unit'][idx],
            }

            # smpl params to transfer translationa nd glob to camera 
            gg = axis_angle_to_matrix(torch.from_numpy(data['smplx_global_orient'])[idx,:,:,:]).numpy()
            tt = np.array(data['smplx_transl'][idx,:,:])
            pelvis = np.zeros((2,3))
            for curr_human_idx in range(2):
                # curr_gender = meta_data['genders'][gender_idx]
                curr_betas = torch.from_numpy(data['smplx_betas'][idx, [curr_human_idx], :])
                pelvis[curr_human_idx,:] = self.body_models_hi4d(
                    betas = curr_betas[:,:10]
                ).joints[0, 0, :].detach().numpy()

            for cam_id, cam_data in data['image_data'][idx].items():
                if self.load_from_scratch_single_camera:
                    if cam_id != '4':
                        continue
                data_item_cam = data_item.copy()

                # bring people into camera coordinate system  
                curr_camera_data = get_current_cam(camera_data, cam_id)           
                # use aitviewer to figur eout camera transformation
                # https://github.com/eth-ait/aitviewer/blob/main/aitviewer/scene/camera.py#L494
                RR_hi4d = curr_camera_data['extrinsics'][0,:,:3]
                # RR180x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                RR = RR_hi4d.T.copy()
                RR[:,1:] *= -1
                # RR = RR180x @ RR # bring to model coordinate system
                TT = curr_camera_data['extrinsics'][:,:,3]  
                global_orient_cam = np.matmul(gg.transpose(0, 1, 3, 2), RR.transpose(1 ,0)).transpose(0,1,3,2)
                data_item_cam['global_orient_cam_smplx'] = matrix_to_axis_angle(torch.from_numpy(global_orient_cam)).float().numpy()[:,0,:]

                # pelv = np.array(smpl_params['pelvis'][:,[array_idx],:])
                transl_cam = np.matmul((tt + pelvis) - TT, np.transpose(RR)) - pelvis
                data_item_cam['transl_cam_smplx'] = torch.from_numpy(transl_cam).float().numpy()

                # save Hi4D mesh to file
                # for meshidx in range(2): 
                #     verts = self.body_models_hi4d(
                #         global_orient = torch.from_numpy(data_item_cam['global_orient_cam_smplx'])[[meshidx]],
                #         body_pose = torch.from_numpy(data_item_cam['body_pose_smplx'])[[meshidx]],
                #         betas = torch.from_numpy(data_item_cam['betas_smplx'])[[meshidx]],
                #         transl = torch.from_numpy(data_item_cam['transl_cam_smplx'])[[meshidx]],
                #     ).vertices.detach().numpy()
                #     mesh = trimesh.Trimesh(verts[0], self.body_models_hi4d.faces, process=False)
                #     _ = mesh.export(f'outdebug/hi4d_mesh/gt_{subject}_{action}_{idx}_{cam_id}_{meshidx}.obj')

                cc_image_name_raw = [x for x in cam_data.keys()][0]
                data_item_cam['imgname'] = f'{subject}_{action}_{cam_id}_{cc_image_name_raw}'
                data_item_cam['imgpath'] = os.path.join(data['action_path'][idx], f'images/{cam_id}/{cc_image_name_raw}.jpg')
                cc = cam_data[cc_image_name_raw]
                human_data = self.concatenate_dicts(cc[0], cc[1])
                
                # check if all information was detected
                data_item_cam['information_missing'] = False
                for x in ['openpose_human_idx', 'bev_human_idx', 'vitpose_human_idx', 'vitposeplus_human_idx']:
                    if np.any(human_data[x] == -1):
                        data_item_cam['information_missing'] = True
                
                human_data = self.process_bev(human_data, (900, 900))

                # save BEV meshes to file
                # for meshidx in range(2):
                #     verts = self.body_models_hi4d(
                #         global_orient = human_data['bev_smplx_global_orient'][[meshidx]],
                #         body_pose = human_data['bev_smplx_body_pose'][[meshidx]],
                #         betas = torch.from_numpy(human_data['bev_smplx_betas'][[meshidx]]),
                #         transl = human_data['bev_smplx_transl'][[meshidx]],
                #     ).vertices.detach().numpy()
                #     mesh = trimesh.Trimesh(verts[0], self.body_models_hi4d.faces, process=False)
                #     _ = mesh.export(f'outdebug/hi4d_mesh/bev_{subject}_{action}_{idx}_{cam_id}_{meshidx}.obj')

                data_item_cam.update(human_data)        
                data_out.append(data_item_cam)

        return data_out

    def load(self, load_from_scratch=False, allow_missing_information=True, processed_fn_ext='.pkl'):
        
        self.load_params_only = False # load only smpl model params, not vertices etc.
        if '_diffusion' in processed_fn_ext:
            self.load_params_only = True
            guru.info('Store only SMPL params in pickle file, not vertices etc.')

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
            guru.info(f'Loaded processed data from {processed_data_path}. Num samples: {len(data)}')
        else:            

            guru.info(f'Processing data from {self.original_data_folder}')

            data = []
            # iterate though dataset / images
            #for imgname, anno in tqdm(self.annotation.items()):
            for subject in tqdm(self.subjects):
                subject_folder = osp.join(self.original_data_folder, 'pair'+str(subject))
                for action in tqdm(os.listdir(subject_folder)):
                    if action.startswith("."):
                        continue
                    data += self.load_single_image(subject, action)
                    try: 
                        pass
                        # data += self.load_single_image(subject, action) 
                    except Exception as e:                
                        #if exeption is keyboard interrupt end program
                        if isinstance(e, KeyboardInterrupt):
                            raise e                    
                        else:
                            print(f'Error loading {subject}/{action}')
                            print(f'Exception: {e}')
                            continue

            # save data to processed data folder
            guru.info(f'Saving processed data to {processed_data_path}. Num samples: {len(data)}')
            with open(processed_data_path, 'wb') as f:
                pickle.dump(data, f)
        
        if self.load_single_camera or self.load_unit_glob_and_transl:
            data = [x for x in data if '/4/' in x['imgpath']]

        if not allow_missing_information:
            data = [x for x in data if not x['information_missing']]

        if self.overfit:
            data = data[:self.overfit_num_samples]
        
        guru.info(f'Final number of samples in Hi4D: {len(data)}')

        return data