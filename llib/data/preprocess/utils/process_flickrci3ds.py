import os.path as osp 
import json 
import torch
import numpy as np
import os
import cv2
import pickle
import argparse
from tqdm import tqdm
from llib.utils.image.bbox import iou_matrix
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
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder
        self.imar_vision_datasets_tools_folder =  imar_vision_datasets_tools_folder
        self.split = split
        self.split_folder = 'test' if split == 'test' else 'train'

        self.body_model_type = body_model_type
        self.image_folder = osp.join(self.original_data_folder, self.split_folder, image_folder)
        self.openpose_folder = osp.join(self.processed_data_folder, self.split_folder, openpose_folder)
        self.bev_folder = osp.join(self.processed_data_folder, self.split_folder, bev_folder)
        self.vitpose_folder = osp.join(self.processed_data_folder, self.split_folder, vitpose_folder)
        self.vitposeplus_folder = osp.join(self.processed_data_folder, self.split_folder, vitposeplus_folder)

        # Flickr human-human-contact annotations
        annotation_fn = osp.join(
            self.original_data_folder, self.split_folder, 'interaction_contact_signature.json')
        self.annotation = json.load(open(annotation_fn, 'r'))

        # correspondence openpose / vitpose / bev
        self.correspondence_fn = osp.join(self.processed_data_folder, self.split_folder, 'correspondence.pkl')
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

    def load_single_image(self, imgname, annotation):
        
        # annotation / image paths
        img_path = osp.join(self.image_folder, f'{imgname}.png')
        bev_path = osp.join(self.bev_folder, f'{imgname}_0.08.npz')
        openpose_path = osp.join(self.openpose_folder, f'{imgname}.json')
        vitpose_path = osp.join(self.vitpose_folder, f'{imgname}_keypoints.json')
        vitposeplus_path = osp.join(self.vitposeplus_folder, f'{imgname}.pkl')

        # load each annotation file
        IMG = cv2.imread(img_path)
        height, width, _ = IMG.shape

        # load bev
        if os.path.exists(bev_path):
            bev_data = np.load(bev_path, allow_pickle=True)['results'][()]
        else:
            bev_data = None
        
        # load keypoints
        op_data = json.load(open(openpose_path, 'r'))['people']
        vitpose_data = json.load(open(vitpose_path, 'r'))['people']
        vitposeplus_data = pickle.load(open(vitposeplus_path, 'rb'))

        ################ Match HHC annotation with OpenPose and ViT keypoints #################
        op_bbox = self.bbox_from_openpose(op_data, kp_key='pose_keypoints_2d') 
        ci_bbox = np.array(annotation['bbxes'])
        ciop_iou_matrix, ciop_iou_best_match = iou_matrix(ci_bbox, op_bbox)

        ################ output #################        
        self.output[imgname] = {
            'ci_sign_openpose_bbox_iou_best_match': ciop_iou_best_match,
            'img_height': height,
            'img_width': width,
        }

        # get all person ids with contact labels
        unique_person_ids = [x['person_ids'] for x in  annotation['ci_sign']]
        unique_person_ids = np.unique(np.array(unique_person_ids).flatten())

        for bbox_id, bbox in enumerate(annotation['bbxes']):

            if bbox_id not in unique_person_ids:
                continue

            temp = {}

            # match the flickr bbox with the openpose bbox
            op_human_idx = ciop_iou_best_match[bbox_id]
    
            bev_human_idx = self.correspondence[imgname]['bev']['best_match'][op_human_idx] if \
                'bev' in self.correspondence[imgname] else -1
            vitpose_human_idx = self.correspondence[imgname]['vitpose']['best_match'][op_human_idx] if \
                'vitpose' in self.correspondence[imgname] else -1
            vitposeplus_human_idx = self.correspondence[imgname]['vitposeplus']['best_match'][op_human_idx] if \
                'vitposeplus' in self.correspondence[imgname] else -1

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
            temp['flickr_bbox'] = bbox
            temp['openpose_human_idx'] = op_human_idx
            temp['bev_human_idx'] = bev_human_idx
            temp['vitpose_human_idx'] = vitpose_human_idx
            temp['vitposeplus_human_idx'] = vitposeplus_human_idx

            for k, v in bev_params.items(): 
                temp[k] = v

            temp['openpose'] = op_kpts
            temp['vitpose'] = vitpose_kpts
            temp['vitposeplus'] = vitposeplus_kpts

            # temp to arrays 
            for k, v in temp.items():
                temp[k] = np.array(v)
            
            # add to full clean dataset
            self.output[imgname][bbox_id] = temp
        


    def process(self):

        processed_data_path = osp.join(
            self.processed_data_folder, self.split_folder, f'processed.pkl'
        )

        guru.info(f'Processing data from {self.original_data_folder}')

        # iterate though dataset / images
        for imgname, anno in tqdm(self.annotation.items()):
            self.load_single_image(imgname, anno)

        # save data to processed data folder
        with open(processed_data_path, 'wb') as f:
            pickle.dump(self.output, f)


### PROCESS PGT FITS FOR RELEASE ###

class BEVTransl():
    
    def __init__(
        self,
        original_data_folder,
        processed_data_folder,
        split='train',
        **kwargs,
    ):  

        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder
        self.split = split
        
        # validation data/images must be loaded from training folder
        self.split_folder = 'test' if split == 'test' else 'train'

        # load processed data (bev, keypoints, etc.)
        processed_fn = osp.join(
            self.processed_data_folder, self.split_folder, 'processed.pkl'
        )
        self.processed = pickle.load(open(processed_fn, 'rb'))

        annotation_fn = osp.join(
            self.original_data_folder, self.split_folder, 'interaction_contact_signature.json'
        )
        self.annotation = json.load(open(annotation_fn, 'r'))

        # convert smpl betas to smpl-x betas 
        self.shape_converter_smpla = ShapeConverter(inbm_type='smpla', outbm_type='smplxa')

        # create body model to get bev root translation from pose params
        self.body_model = self.shape_converter_smpla.outbm

    def process_bev(self, data):

            body_pose = data['bev_smpl_body_pose'][:,:63]
            global_orient = data['bev_smpl_global_orient']
            betas = data['bev_smplx_betas']
            scale = data['bev_smplx_scale']
            
            bev_cam_trans = torch.from_numpy(data['bev_cam_trans'])

            transls = []
            for idx in range(2):

                h_global_orient = torch.from_numpy(global_orient[[idx]])
                
                h_body_pose = torch.from_numpy(body_pose[[idx]])

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
                transls.append([transl])

            return transls


    def concatenate_dicts(self, x, y):
        concatenated_dict = {}
        for key in x.keys():
            concatenated_dict[key] = np.stack((x[key], y[key]), axis=0)
        return concatenated_dict

    def get_transl(self, imgname, case_ci):

        processed = self.processed[imgname]
        
        # load contact annotations
        p0id, p1id = case_ci['person_ids']        

        ################ load the two humans in contact #################
        human_data = self.concatenate_dicts(processed[p0id], processed[p1id])
        human_data = self.process_bev(human_data)

        th0, th1 = human_data[0][0][0][0], human_data[1][0][0][0]
        h0id = 0 if th0 <= th1 else 1
        h1id = 1-h0id
        new_human_idxs = [h0id, h1id]
        return new_human_idxs

      

def process_pgt_fits(original_data_folder, processed_data_folder, pgt_fits_folder,
                    split='train'):

    pgt_folder = f'{processed_data_folder}/{split}/{pgt_fits_folder}/results/'
    output_fn = f'{processed_data_folder}/{split}/processed_pseudogt_fits.pkl'

    #bevtransl = BEVTransl(
    #    original_data_folder, processed_data_folder, split=split,
    #)

    #def concatenate_dicts(x, y):
    #    concatenated_dict = {}
    #    for key in x.keys():
    #        concatenated_dict[key] = np.stack((x[key], y[key]), axis=0)
    #    return concatenated_dict

    output = {}

    for fn in tqdm(os.listdir(pgt_folder)):
        fnkey = fn.split('.')[0]
        #imgname_split = fnkey.split("_")
        #imgname = "_".join(imgname_split[:-1])
        #case_ci_idx = int(imgname_split[-1])
        #case_ci = bevtransl.annotation[imgname]['ci_sign'][case_ci_idx]
        # we keep the same order in the new PGT fits
        #bev_transl_human_idxs = bevtransl.get_transl(imgname, case_ci)
        pkl_path = osp.join(pgt_folder, fn)
        #data_out = {}
        with open(pkl_path, 'rb') as f:
            data_out = pickle.load(f)
        h0i, h1i = 0,1 #bev_transl_human_idxs 
        #humans = concatenate_dicts(data[f'h{h0i}'], data[f'h{h1i}'])
        #tx, ty, tz = humans.pop('translx'), humans.pop('transly'), humans.pop('translz')
        #humans['transl'] = np.concatenate((tx, ty, tz), axis=1)
        #data_out['humans'] = humans
        #data_out['cam'] = data['cam']
        data_out['preprocess_human_idxs'] = [h0i, h1i] #bev_transl_human_idxs
        output[fnkey] = data_out

    # save output
    pickle.dump(output, open(output_fn, 'wb'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-full-processing', action='store_true')
    parser.add_argument('--run-pgt-processing', action='store_true')
    args = parser.parse_args()
   
    original_data_folder = 'datasets/original/FlickrCI3D_Signatures'
    processed_data_folder = 'datasets/processed/FlickrCI3D_Signatures'
    imar_vision_datasets_tools_folder = 'essentials/imar_vision_datasets_tools'


    for split_folder in ['train', 'test']:
        if args.run_full_processing:
            # read bev, openpose, vitpose etc. and save to [split-folder]/processed.pkl
            flickr_data = FlickrCI3D_Signatures(
                original_data_folder,
                processed_data_folder,
                imar_vision_datasets_tools_folder,
                image_folder='images',
                bev_folder='bev',
                openpose_folder='keypoints/openpose',
                split=split_folder,
                body_model_type='smplx',
                vitpose_folder='keypoints/vitpose',
                vitposeplus_folder='keypoints/vitposeplus',
            )
        
            flickr_data.process()
        
        if args.run_pgt_processing:
            # read pgt fits from disk and save to [split-folder]/processed_pseudogt_fits.pkl
            process_pgt_fits(
                original_data_folder,
                processed_data_folder,
                pgt_fits_folder='hhcs_opti/run4_20231015',
                split=split_folder,
            )