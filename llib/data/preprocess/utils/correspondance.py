import numpy as np
import torch
from llib.utils.image.bbox import iou_matrix
from llib.utils.keypoints.matching import keypoint_cost_matrix
from llib.bodymodels.utils import smpl_to_openpose
import os.path as osp
import cv2 
import json 
import pickle
import os
from tqdm import tqdm
import argparse


class CorrespondenceSolver():

    def __init__(
        self,
        output_folder='',
        image_folder='images',
        bev_folder='bev',
        openpose_folder='keypoints/openpose',
        vitpose_folder='keypoints/vitpose',
        vitposeplus_folder='keypoints/vitposeplus',
        body_model_type='smpl',
        openpose_format='coco25',
        bev_extension='_0.08.npz',
        unique_best_matches=False,
        reference_method='openpose',
        reference_data=None,
        reference_data_ext='',
        testing_methods=['bev', 'vitpose', 'vitposeplus'],
        conf_thres=0.6,
        min_conf_crit_count=1,
    ):  
        """
        Solve correspondance between people starting from OpenPose.
        From OpenPose keypoints, compute costs with other keypoints
        or projected joints and solve correspondance with Hungarian
        matching.
        """

        # input folder
        self.openpose_folder = openpose_folder
        self.bev_folder = bev_folder
        self.vitpose_folder = vitpose_folder
        self.vitposeplus_folder = vitposeplus_folder 
        self.body_model_type = body_model_type
        self.image_folder = image_folder
        self.bev_extention = bev_extension
        self.unique_best_matches = unique_best_matches

        # the cost matrix is computed between the reference method 
        # and all the testing methods
        self.reference_method = reference_method
        self.reference_data = reference_data
        self.reference_data_ext = reference_data_ext
        self.testing_methods = testing_methods

        # thresholds for keypoints detection
        self.conf_thres = conf_thres
        self.min_conf_crit_count = min_conf_crit_count

        if self.reference_method == 'ground_truth':
            assert reference_data is not None, \
                'ground truth data must be provided if reference method is ground truth'

        # smpl joints to openpose joints for BEV / Openpose matching
        self.smpl_to_op_map = smpl_to_openpose(
            model_type='smpl', use_hands=False, use_face=False,
            use_face_contour=False, openpose_format=openpose_format
        )

        # output
        self.output_folder = output_folder
        self.output = {}

    def load_image(self, imgname_fn):
        image_path = osp.join(self.image_folder, f'{imgname_fn}')
        image = cv2.imread(image_path)
        return image

    def load_openpose(self, imgname):
        openpose_path = osp.join(self.openpose_folder, f'{imgname}.json')
        op_data = json.load(open(openpose_path, 'r'))['people']
        body_keypoints = [np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in op_data]
        return body_keypoints

    def load_vitpose(self, imgname):
        vitpose_path = osp.join(self.vitpose_folder, f'{imgname}_keypoints.json')
        vitpose_data = json.load(open(vitpose_path, 'r'))['people']
        body_keypoints = [np.array(x['pose_keypoints_2d']).reshape(-1,3) for x in vitpose_data]
        return body_keypoints

    def load_vitposeplus(self, imgname):
        vitposeplus_path = osp.join(self.vitposeplus_folder, f'{imgname}.pkl')
        vitposeplus_data = pickle.load(open(vitposeplus_path, 'rb'))

        body_keypoints = []
        for item in vitposeplus_data:
            output = np.zeros([25, 3])
            output[[0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]] = item['keypoints'][:17]
            output[19:25] = item['keypoints'][17:23]
            body_keypoints.append(output)
        
        return body_keypoints

    def load_bev(self, imgname):
        bev_path = osp.join(self.bev_folder, f'{imgname}{self.bev_extention}')
        bev_data = np.load(bev_path, allow_pickle=True)['results'][()]
        body_keypoints = [
            np.concatenate((x.reshape(-1,2)[self.smpl_to_op_map,:], 
            np.ones((25, 1))), axis=1) for x in bev_data['pj2d_org']
        ]
        return body_keypoints

    def match(self, imgname, imgnorm, openpose, methodname):
        # compute cost between openpose and method_name
        # to access the best match for a OpenPose detetion at index op_human_idx use
        # methodname_human_idx = best_match[op_human_idx]

        data = eval(f'self.load_{methodname}')(imgname)
        cost_matrix, best_match = keypoint_cost_matrix(
            kpts1=openpose, 
            kpts2=data, 
            norm=imgnorm,
            unique_best_matches=self.unique_best_matches,
            conf_thres=self.conf_thres, 
            min_conf_crit_count=self.min_conf_crit_count
        )
        return cost_matrix, best_match

    def plot_over_image(self, frame, points_2d=[], path_to_write=None):
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

    def match_all(self, imgname_fn):
            image = self.load_image(imgname_fn)
            imgname = imgname_fn.split('.')[0]

            self.output[imgname] = {}

            imgnorm = max(image.shape[0], image.shape[1])
            if self.reference_method == 'openpose':
                reference_kpts = self.load_openpose(imgname)
            else:
                reference_kpts = self.reference_data[imgname + self.reference_data_ext]

            for methodname in self.testing_methods:
                try:
                    cost_matrix, best_match = self.match(imgname, imgnorm, reference_kpts, methodname)
                    self.output[imgname][methodname] = {
                        'cost_matrix': cost_matrix,
                        'best_match': best_match,
                    }
                    # visualize results
                    #kpts =  eval(f'self.load_{methodname}')(imgname)
                    #self.plot_over_image(image, 
                    # points_2d=[reference_kpts[0], reference_kpts[1], kpts[best_match[0]], kpts[best_match[1]]], 
                    # path_to_write=f'outdebug/{imgname}_{methodname}.png'
                    #)
                except:
                    print(f'Error in {imgname_fn} {methodname}')

    def save_output(self, output_fn='correspondence.pkl'):
        output_path = osp.join(self.output_folder, output_fn)
        pickle.dump(self.output, open(output_path, 'wb'))

    def process_folder(self, save_output=True, output_fn='correspondence.pkl'):
        for imgname in tqdm(os.listdir(self.image_folder)):
            self.match_all(imgname)
        
        if save_output:
            self.save_output(output_fn=output_fn)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='FlickrCI3D_Signatures')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--single_camera', action='store_true', help='process single camera (id 4) for Hi4D')
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    split = args.split
    
    if args.dataset_name == 'FlickrCI3D_Signatures':
        processed_dataset_folder = f'datasets/processed/{dataset_name}/{split}'
        original_dataset_folder = f'datasets/original/{dataset_name}/{split}'

        solver = CorrespondenceSolver(
            output_folder=processed_dataset_folder,
            image_folder=osp.join(original_dataset_folder, 'images'),
            bev_folder=osp.join(processed_dataset_folder, 'bev'),
            openpose_folder=osp.join(processed_dataset_folder, 'keypoints/openpose'),
            vitpose_folder=osp.join(processed_dataset_folder, 'keypoints/vitpose'),
            vitposeplus_folder=osp.join(processed_dataset_folder, 'keypoints/vitposeplus'),
            body_model_type='smpl',
            openpose_format='coco25',
            reference_method='openpose',
            testing_methods=['bev', 'vitpose', 'vitposeplus'],
        ).process_folder()
    elif args.dataset_name == 'CHI3D':
        processed_dataset_folder_base = f'datasets/processed/{dataset_name}/{split}'
        original_dataset_folder_base = f'datasets/original/{dataset_name}/{split}'
        ground_truth_joints_2d = pickle.load(
            open(osp.join(processed_dataset_folder_base, 'images_contact_projected_joints_2d.pkl'), 'rb'))
        all_output = {}

        for subject in os.listdir(original_dataset_folder_base):
            processed_dataset_folder = osp.join(processed_dataset_folder_base, subject)
            original_dataset_folder = osp.join(original_dataset_folder_base, subject)

            solver = CorrespondenceSolver(
                output_folder=processed_dataset_folder,
                image_folder=osp.join(processed_dataset_folder, 'images_contact'),
                bev_folder=osp.join(processed_dataset_folder, 'bev'),
                openpose_folder=osp.join(processed_dataset_folder, 'images_contact_openpose/keypoints'),
                vitpose_folder=osp.join(processed_dataset_folder, 'images_contact_vitpose'),
                vitposeplus_folder=osp.join(processed_dataset_folder, 'images_contact_vitposeplus'),
                body_model_type='smplx',
                openpose_format='coco25',
                bev_extension='__2_0.08.npz',
                unique_best_matches=True,
                reference_method='ground_truth',
                reference_data=ground_truth_joints_2d['openpose_coco25'][subject],
                testing_methods=['bev', 'vitpose', 'vitposeplus', 'openpose'],
                conf_thres=0.6,
                min_conf_crit_count=4,
            )
            
            solver.process_folder(save_output=False)
            all_output[subject] = solver.output
    elif args.dataset_name == 'Hi4D':
        processed_dataset_folder_base = f'datasets/processed/{dataset_name}'
        original_dataset_folder_base = f'datasets/original/{dataset_name}'
        ground_truth_joints_2d = pickle.load(
            open(osp.join(processed_dataset_folder_base, 'images_contact_projected_joints_2d.pkl'), 'rb'))
        all_output = {}
        correspondence_fn = 'correspondence.pkl' if not args.single_camera \
            else 'correspondence_single_camera.pkl'

        for subject in os.listdir(original_dataset_folder_base):
            all_output[subject] = {}
            if not subject.startswith('pair'):
                continue
            actions = os.listdir(osp.join(original_dataset_folder_base, subject))
            for action in actions:
                all_output[subject][action] = {}
                if action.startswith('.'):
                    continue

                cam_img_folder = osp.join(original_dataset_folder_base, subject, action, 'images')

                for cam in os.listdir(cam_img_folder):
                    if cam.startswith('.'):
                        continue
                    if args.single_camera and cam != '4':
                        continue
                    processed_dataset_folder = osp.join(processed_dataset_folder_base, subject, action)
                    original_dataset_folder = osp.join(original_dataset_folder_base, subject, action)

                    solver = CorrespondenceSolver(
                        output_folder=processed_dataset_folder,
                        image_folder=osp.join(original_dataset_folder, 'images', cam),
                        bev_folder=osp.join(processed_dataset_folder, 'bev', cam),
                        openpose_folder=osp.join(processed_dataset_folder, 'openpose/keypoints', cam),
                        vitpose_folder=osp.join(processed_dataset_folder, 'keypoints/vitpose', cam),
                        vitposeplus_folder=osp.join(processed_dataset_folder, 'keypoints/vitposeplus', cam),
                        body_model_type='smpl',
                        openpose_format='coco25',
                        bev_extension='__2_0.08.npz',
                        unique_best_matches=True,
                        reference_method='ground_truth',
                        reference_data=ground_truth_joints_2d['openpose_coco25'][subject][action][cam],
                        reference_data_ext='',
                        testing_methods=['bev', 'vitposeplus', 'vitpose', 'openpose'],
                        conf_thres=0.6,
                        min_conf_crit_count=4,
                    )
                    
                    solver.process_folder(save_output=False)
                    all_output[subject][action][cam] = solver.output

        # save output
        output_path = osp.join(processed_dataset_folder_base, correspondence_fn)
        pickle.dump(all_output, open(output_path, 'wb'))
    else:
        raise NotImplementedError