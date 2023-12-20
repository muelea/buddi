import os.path as osp 
import json 
import torch
import numpy as np
import os
import cv2
import math
import pickle
import smplx
from tqdm import tqdm
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
        load_from_scratch_single_camera: bool = False,
        load_contact_frame_only=True,
        load_unit_glob_and_transl=False,
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder
        self.imar_vision_datasets_tools_folder =  imar_vision_datasets_tools_folder
        self.split = split
        self.load_single_camera = load_single_camera
        self.load_unit_glob_and_transl = load_unit_glob_and_transl
        self.load_from_scratch_single_camera = load_from_scratch_single_camera
        self.load_contact_frame_only = load_contact_frame_only
        
        # validation data/images must be loaded from training folder
        self.split_folder = 'test' if split == 'test' else 'train'
        if self.split_folder == 'train':
            trainval_fn = osp.join(self.processed_data_folder, self.split_folder, 'train_val_split.npz')
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
        
        # load processed data (bev, keypoints, etc.)
        processed_fn = osp.join(
            self.processed_data_folder, self.split_folder, 'images_contact_processed.pkl'
        )
        self.processed = pickle.load(open(processed_fn, 'rb'))

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

    def read_cam_params(self, cam_path):
        with open(cam_path) as f:
            cam_params = json.load(f)
            for key1 in cam_params:
                for key2 in cam_params[key1]:
                    cam_params[key1][key2] = np.array(cam_params[key1][key2]) 
        return cam_params

    def concatenate_dicts(self, x, y):
        concatenated_dict = {}
        for key in x.keys():
            try:
                concatenated_dict[key] = np.stack((x[key], y[key]), axis=0)
            except:
                import ipdb; ipdb.set_trace()
        return concatenated_dict
    
    def load_smpl_data(self, smpl_fn, frame_ids=[85]):

        smplx_data = json.load(open(smpl_fn, 'r'))
        for k, v in smplx_data.items():
            smplx_data[k] = np.array(v)

        num_frames = len(frame_ids)
        ################ load the two humans in contact ########
        betas = torch.from_numpy(np.array(smplx_data['betas'][:,frame_ids,:])).float()
        betas_with_scale = torch.cat((betas, torch.zeros(2, num_frames, 1)), dim=-1)

        # create rotation matrix for 90 degree x axis rotation 
        RR = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        # apply rotation to smpl data
        global_orient = np.array(smplx_data['global_orient'][:,frame_ids,:])
        global_orient_unit = np.matmul(global_orient.transpose(0, 1, 2, 4, 3), RR.transpose(1 ,0)).transpose(0,1,2,4,3)

        # translation
        init_transl = np.array(smplx_data['transl'][:,frame_ids,:])
        new_transl = np.zeros_like(init_transl)
        all_pelvis = np.zeros_like(init_transl)
        for human_idx in [0, 1]:
            for array_idx in range(init_transl.shape[1]):
                # bring mesh into same position as BEV estimate 
                # get pelvis of person with markers
                pelvis = self.body_model(betas=betas_with_scale[human_idx, array_idx, :][None]).joints[:, 0, :].detach().numpy()
                all_pelvis[human_idx, array_idx] = pelvis[0]
                # apply rotation to smpl data
                transl = np.array(init_transl[human_idx,array_idx])[None]
                transl_unit = np.matmul(transl + pelvis, np.transpose(RR)) - pelvis
                new_transl[human_idx, array_idx] = transl_unit

        new_transl = torch.from_numpy(new_transl).float()

        params = {
            'global_orient': matrix_to_axis_angle(torch.from_numpy(global_orient_unit)).float()[:,:,0,:],
            'transl': new_transl,
            'global_orient_orig': matrix_to_axis_angle(torch.from_numpy(global_orient)).float()[:,:,0,:],
            'transl_orig': torch.from_numpy(init_transl).float(),
            'global_orient_orig_matrix': torch.from_numpy(global_orient).float()[:,:,0,:],            
            'body_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['body_pose'][:,frame_ids,:]))).view(2, num_frames, -1).float(), 
            'shape': betas_with_scale,
            'betas': betas,
            'scale': torch.zeros(2, num_frames, 1),
            'left_hand_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['left_hand_pose'][:,frame_ids,:]))).float().view(2, num_frames, -1)[:,:,:6],
            'right_hand_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['right_hand_pose'][:,frame_ids,:]))).float().view(2, num_frames, -1)[:,:,:6], 
            'jaw_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['jaw_pose'][:,frame_ids,:]))).float()[:,:,0,:], 
            'leye_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['leye_pose'][:,frame_ids,:]))).float()[:,:,0,:], 
            'reye_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['reye_pose'][:,frame_ids,:]))).float()[:,:,0,:], 
            'expression': torch.from_numpy(np.array(smplx_data['expression'][:,frame_ids,:])).float(),
            'pelvis': torch.from_numpy(all_pelvis).float(),
        }

        # params[f'vertices'] = np.zeros((2, num_frames, 10475, 3)) #body.vertices.detach().to('cpu').numpy()[0]
        # params[f'joints'] = np.zeros((2, num_frames, 127, 3)) #body.joints.detach().to('cpu').numpy()[0]
        # params[f'bev_keypoints'] = np.zeros((2, num_frames, 127, 2))
        # params[f'bev_orig_vertices'] = np.zeros((2, num_frames, 6890, 3))
        # params[f'op_keypoints'] = np.zeros((2, num_frames, 25, 3))
        # params[f'vitpose_keypoints'] = np.zeros((2, num_frames, 25, 3))

        # for human_idx in [0,1]:
        #     for array_idx in range(init_transl.shape[1]):
        #         params_for_smpl = {} 
        #         for key, val in params.items():
        #             if 'betas' in key:
        #                 continue
        #             if 'scale' in key:
        #                 continue
        #             if 'shape' in key:
        #                 params_for_smpl['betas'] = val[human_idx, array_idx, :][None]
        #             else:
        #                 params_for_smpl[key] = val[human_idx, array_idx, :][None]

        #         body = self.body_model(**params_for_smpl)

        #         # save mesh for debugging 
        #         #import trimesh
        #         #mm = trimesh.Trimesh(body.vertices.detach().to('cpu').numpy()[0],self.body_model.faces)
        #         #mm.export(f'meshes/m{human_idx}_63_transformed_[1, 0, 0], [0, 0, -1], [0, 1, 0].obj')

        #         params[f'vertices'][human_idx, array_idx] = body.vertices.detach().to('cpu').numpy()[0]
        #         params[f'joints'][human_idx, array_idx] = body.joints.detach().to('cpu').numpy()[0]
        #         params[f'bev_keypoints'][human_idx, array_idx] = np.zeros((127, 2))
        #         params[f'bev_orig_vertices'][human_idx, array_idx] = np.zeros((6890, 3))
        #         params[f'op_keypoints'][human_idx, array_idx] = np.zeros((25, 3))
        #         params[f'vitpose_keypoints'][human_idx, array_idx] = np.zeros((25, 3))

        return params

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

    def get_camera_smplx_params(self, smplx_data, human_params, cam_params, human_idx, frame_id):

        # get pelvis of person with markers
        betas = human_params[human_idx]['betas']
        pelvis = self.body_model_conversion(betas=betas).joints[:, 0, :].detach().numpy()

        # camera params
        RR = cam_params['extrinsics']['R']
        TT = cam_params['extrinsics']['T']

        global_orient = np.array(smplx_data['global_orient'][human_idx][frame_id])
        new_global_orient = np.matmul(global_orient.transpose(0, 2, 1), RR.transpose(1 ,0)).transpose(0,2,1)

        transl = np.array(smplx_data['transl'][human_idx][frame_id])[None]
        new_transl = np.matmul(transl + pelvis - TT, np.transpose(RR)) - pelvis

        return new_global_orient, new_transl

    def load_single_image(self, subject, action, annotation):

        orig_subject_folder = osp.join(self.original_data_folder, self.split_folder, subject)
        processed_subject_folder = osp.join(self.processed_data_folder, self.split_folder, subject)

        frame_ids = list(np.arange(annotation['fr_id'], annotation['start_fr'], -5)[::-1]) + \
                    list(np.arange(annotation['fr_id'], annotation['end_fr']-1, 5)[1:])
        if len(frame_ids) == 0:
            frame_ids = [annotation['fr_id']]

        # get camera parameters 
        cameras = os.listdir(osp.join(orig_subject_folder, 'camera_parameters'))

        # load SMPL params
        smpl_path = f'{orig_subject_folder}/smplx/{action}.json'
        smpl_params = self.load_smpl_data(smpl_path, frame_ids)

        ################ load the two humans in contact #################
        frame_data = {}
        region_id = annotation[f'{self.body_model_type}_signature']['region_id']        
        contact_map = self.contact_zeros.clone()
        for rid in region_id:
            contact_map[rid[0], rid[1]] = True

        data = []
        for array_idx, frame_id in enumerate(frame_ids):
            for cam in cameras:
                if self.load_from_scratch_single_camera:
                    if cam != '58860488':
                        continue
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
                    'img_out_fn': f'{subject}_{img_name}_0', #_0 since contact index is 0 bec. only two people on image
                    'img_height': height,
                    'img_width': width,
                    'cam_transl': cam_transl,
                    'cam_rot': cam_rot,
                    'fl': focal_length_px,
                    'afov_horizontal': self.BEV_FOV,
                    'information_missing': False,
                    'is_contact_frame': False
                }
                
                for k, v in smpl_params.items():
                    data_item[f'{k}'] = v[:, [array_idx], :]

                # bring people into camera coordinate system                     
                gg = np.array(smpl_params['global_orient_orig_matrix'][:,[array_idx],:,:])
                global_orient_cam = np.matmul(gg.transpose(0, 1, 3, 2), RR.cpu().numpy().transpose(1 ,0)).transpose(0,1,3,2)
                data_item['global_orient_cam'] = matrix_to_axis_angle(torch.from_numpy(global_orient_cam)).float()

                tt = np.array(smpl_params['transl_orig'][:,[array_idx],:])
                pelv = np.array(smpl_params['pelvis'][:,[array_idx],:])
                transl_cam = np.matmul((tt + pelv) - TT.cpu().numpy(), np.transpose(RR.cpu().numpy())) - pelv
                data_item['transl_cam'] = torch.from_numpy(transl_cam).float()
 
                if frame_id == annotation['fr_id']:
                    cc = self.processed[subject][img_name]
                    human_data = self.concatenate_dicts(cc[0], cc[1])

                    # check if all detections (keypoints, bev) are available 
                    for x in ['openpose_human_idx', 'bev_human_idx', 'vitpose_human_idx', 'vitposeplus_human_idx']:
                        if np.any(human_data[x] == -1):
                            data_item['information_missing'] = True

                    # Update the data_item with the human data
                    human_data = self.process_bev(human_data, (900, 900))
                    data_item.update(human_data)
                    data_item['contact_map'] = contact_map
                    data_item['hhc_contacts_region_ids'] = region_id
                    data_item['is_contact_frame'] = True

                data.append(data_item)
        
        return data

    def load(self, load_from_scratch=False, allow_missing_information=True, processed_fn_ext='.pkl'):

        processed_data_path = osp.join(
            self.processed_data_folder, f'{self.split}{processed_fn_ext}'
        )

        # load data if it exists, otherwise process it
        if osp.exists(processed_data_path) and not load_from_scratch:      
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
            num_samples = len(data)
            guru.info(f'Loading processed data from {processed_data_path}. Num samples {num_samples}.')   
        else:            

            guru.info(f'Processing data from {self.original_data_folder}')

            data = []
            # iterate though dataset / images
            #for imgname, anno in tqdm(self.annotation.items()):
            for subject in tqdm(self.subjects):
                annotation_fn =  osp.join(
                    self.original_data_folder, self.split_folder, subject, 'interaction_contact_signature.json'
                )
                annotation = json.load(open(annotation_fn, 'r'))

                for action, anno in tqdm(annotation.items()):
                    data += self.load_single_image(subject, action, anno)

            # save data to processed data folder
            with open(processed_data_path, 'wb') as f:
                pickle.dump(data, f)

        if self.load_single_camera or self.load_unit_glob_and_transl:
            data = [x for x in data if '58860488' in x['imgpath']]
        
        if self.load_contact_frame_only:
            data = [x for x in data if x['is_contact_frame']]
        
        if not allow_missing_information:
            data = [x for x in data if not x['information_missing']]
        
        data = [x for x in data if not ( \
                    np.any(x['bev_human_idx'] == -1) \
                    #or np.any(x['openpose_human_idx'] == -1) \
                    #or np.any(x['vitpose_human_idx'] == -1)
        )]
        if self.overfit:
            data = data[:self.overfit_num_samples]

        guru.info(f'Final number of samples in CHI3D: {len(data)}')
        
        return data
