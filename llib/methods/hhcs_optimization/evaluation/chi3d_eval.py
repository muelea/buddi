import argparse
import torch 
import numpy as np
import cv2
import os
from tqdm import tqdm
import os.path as osp
import pickle
import json
from llib.visualization.utils import *
from llib.utils.metrics.build import build_metric
from llib.bodymodels.build import build_bodymodel
from llib.utils.threed.distance import ContactMap
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms import euler_angles_to_matrix
import math
import smplx 
import shutil
import matplotlib.pyplot as plt
from llib.methods.hhcs_optimization.evaluation.utils import *

from llib.utils.threed.distance import pcl_pcl_pairwise_distance
from llib.defaults.main import (
    config as default_config,
    merge as merge_configs
)

ESSENTIALS_HOME = os.environ['ESSENTIALS_HOME']
PROJECT_HOME = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact' #os.environ['HUMANHUMANCONTACT_HOME']
REGION_TO_VERTEX_PATH = osp.join(ESSENTIALS_HOME, 'contact/flickrci3ds_r75_rid_to_smplx_vid.pkl')


J14_REGRESSOR_PATH = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/SMPLX_to_J14.pkl'
J14_REGRESSOR = torch.from_numpy(
    pickle.load(open(J14_REGRESSOR_PATH, 'rb'), encoding='latin1')).to('cuda').float()

# Indices to get the 14 LSP joints from the ground truth SMPL joints
jreg_path = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/J_regressor_h36m.npy'
SMPL_TO_H36M = torch.from_numpy(np.load(jreg_path)).to('cuda').float()
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

# We need this to only evaluate on images that do not miss keypoint/BEV detections
ORIG_DATA_FOLDER = 'datasets/original/CHI3D'
PROCESSED_DATA_FOLDER = 'datasets/processed/CHI3D'

PROCESSED_DATA = pickle.load(open(f'{PROCESSED_DATA_FOLDER}/train/images_contact_processed.pkl', 'rb'))
TRAIN_VAL_SPLIT = np.load(f'{PROCESSED_DATA_FOLDER}/train/train_val_split.npz')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg', 
        type=str, dest='exp_cfgs', nargs='+', default='llib/methods/hhcs_optimization/evaluation/chi3d_eval.yaml', 
        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
        nargs='*', help='The configuration of the Detector') 
    parser.add_argument('--predictions-folder', type=str, 
        default=f'{PROJECT_HOME}/results/HHC/optimization/fit_mocap_flickrci3ds_test_v02')
    parser.add_argument('--eval-split', default='train', type=str, choices=['train', 'val'])
    parser.add_argument('--print_result', action='store_true', default=False, help='Print the result to the console')
    cmd_args = parser.parse_args()

    cfg = merge_configs(cmd_args, default_config)

    return cfg, cmd_args



def get_smplx_pred(human, body_model_smplx=None):
    """
    Returns the SMPL parameters of a human.
    """
    def to_tensor(x):
        return torch.tensor(x).to('cuda')

    params = dict(
        #betas = torch.cat([to_tensor(human[f'betas']),to_tensor(human[f'scale'])], dim=1),
        global_orient = to_tensor(human[f'global_orient']),
        body_pose = to_tensor(human[f'body_pose']),
        betas = to_tensor(human[f'betas']),
        scale = to_tensor(human[f'scale']),
        transl = to_tensor(human[f'transl']),
    )

    verts, joints = None, None
    if body_model_smplx is not None:
        body = body_model_smplx(**params)
        verts = body.vertices.detach() 
        joints = torch.matmul(J14_REGRESSOR, verts)

    return params, verts, joints


def main(cfg, cmd_args):

    PREDICTIONS_FOLDER = cmd_args.predictions_folder

    # cmd args and logging
    subjects = TRAIN_VAL_SPLIT[cmd_args.eval_split]
    split_folder = 'train' if cmd_args.eval_split in ['train', 'val'] else 'test'
    # actions, subjects_ll, cc, img_names = [], [], [], []

    # build metrics 
    contact_metric = build_metric(cfg.evaluation.cmap_iou)
    scale_mpjpe_metric = build_metric(cfg.evaluation.scale_mpjpe)
    mpjpe_metric = build_metric(cfg.evaluation.mpjpe)
    pa_mpjpe_metric = build_metric(cfg.evaluation.pa_mpjpe)
    pairwise_pa_mpjpe_metric = build_metric(cfg.evaluation.pairwise_pa_mpjpe)

    # SMPL model for predictions
    model_folder = osp.join(ESSENTIALS_HOME, 'body_models')
    kid_template = osp.join(model_folder, 'smil/smplx_kid_template.npy')
    bm_smplxa = smplx.create(
        model_path=model_folder, 
        model_type='smplx',
        kid_template_path=kid_template, 
        age='kid',
        batch_size=2
    ).to(cfg.device)

    bm_smplxa = build_bodymodel(
        cfg=cfg.body_model, 
        batch_size=2, 
        device=cfg.device
    )

    # SMPL model for ground truth
    bm_smplx = smplx.create(
        model_path=model_folder, 
        model_type='smplx',
        batch_size=2
    ).to(cfg.device)

    results = ResultLogger(
        method_names=['est'],
        output_fn=f'{PREDICTIONS_FOLDER}/results.pkl'
    )
    results.info = {
        'actions': [],
        'subjects': [],
        'contact_counts': [],
        'img_names': [],
    }

    cmapper = ContactMap(
        region_to_vertex=REGION_TO_VERTEX_PATH,
    )

    ITEMS_TO_EVAL = chi3d_items_for_eval(
        subjects, split_folder, ORIG_DATA_FOLDER, PROCESSED_DATA
    )


    for subject, actions_dict in tqdm(ITEMS_TO_EVAL.items()):
        annotation_fn =  osp.join(
            ORIG_DATA_FOLDER, split_folder, subject, 'interaction_contact_signature.json'
        )
        annotations = json.load(open(annotation_fn, 'r'))

        for action, cameras in actions_dict.items():
            
            orig_subject_folder = osp.join(ORIG_DATA_FOLDER, split_folder, subject)
            processed_subject_folder = osp.join(PROCESSED_DATA_FOLDER, split_folder, subject)
            
            # gt annotation data 
            annotation = annotations[action]
            frame_id = annotation['fr_id']

            # load SMPL params
            smpl_path = f'{orig_subject_folder}/smplx/{action}.json'
            params_gt, verts_gt, joints_gt = chi3d_get_smplx_gt(smpl_path, [frame_id], bm_smplx)
            verts_gt = torch.from_numpy(verts_gt).to(cfg.device).float()

            region_id = annotation[f'smplx_signature']['region_id']        
            gt_contact_map = np.zeros((75, 75)).astype(bool)
            for rid in region_id:
                gt_contact_map[rid[0], rid[1]] = True

            for cam in cameras:
                img_name = f'{subject}_{action}_{frame_id:06d}_{cam}_0'
                img_path = osp.join(processed_subject_folder, 'images_contact', img_name+'.png')
                PRED_ITEM_PATH = f'{PREDICTIONS_FOLDER}/results/{img_name}.pkl'

                # check if item was in processing / optimization batch or if information was missing
                if not os.path.exists(PRED_ITEM_PATH):
                    print('ITEM MISSING', subject, action, frame_id, cam)
                    continue

                # predicted data
                pred_item = pickle.load(open(PRED_ITEM_PATH, 'rb'))
                params_pred, verts_pred, joints_pred = get_smplx_pred(pred_item['humans'], bm_smplxa)

                # ground truth data
                cam_path = f'{orig_subject_folder}/camera_parameters/{cam}/{action}.json'
                cam_params = chi3d_read_cam_params(cam_path)
                verts_gt_camera =  chi3d_verts_world2cam(verts_gt, cam_params)

                #print(cam_params['extrinsics'])
                #print(pred_item['cam'], pred_item['humans']['transl'])
                # for iii in [0,1]:
                    # save_mesh(verts_gt_camera[iii].cpu().numpy(), bm_smplx.faces, f'outdebug/chi3d_val/{subject}_{action}_{cam}_test_mesh_gt_cam_{iii}.ply')
                    # save_mesh(verts_gt[iii,0].cpu().numpy(), bm_smplx.faces, f'outdebug/chi3d_val/{subject}_{action}_{cam}_test_mesh_gt_{iii}.ply')
                    # save_mesh(verts_pred[iii].cpu().numpy(), bm_smplx.faces, f'outdebug/chi3d_val/{subject}_{action}_{cam}_test_mesh_pred_{iii}.ply')
                    # save_mesh(verts_pred_chi3d_world[iii].cpu().numpy(), bm_smplx.faces, f'outdebug/{subject}_{action}_{cam}_test_mesh_pred_chi3d_w_{iii}.ply')
                    # save_mesh(verts_pred_chi3d_cam[iii].cpu().numpy(), bm_smplx.faces, f'outdebug/{subject}_{action}_{cam}_test_mesh_pred_chi3d_c_{iii}.ply')
                    # save_mesh(verts_gt[iii][0].cpu().numpy(), bm_smplx.faces, f'outdebug/{subject}_{action}_{cam}_test_mesh_gt_{iii}.ply')
                
                est_smplx_vertices = [verts_pred[0], verts_pred[1]]
                est_smplx_joints = [verts2joints(verts_pred[0][None], J14_REGRESSOR), verts2joints(verts_pred[1][None], J14_REGRESSOR)]
                gt_smplx_vertices = [verts_gt_camera[0], verts_gt_camera[1]]
                gt_smplx_joints = [verts2joints(verts_gt_camera[0][None], J14_REGRESSOR), verts2joints(verts_gt_camera[1][None], J14_REGRESSOR)]


                dists = pcl_pcl_pairwise_distance(
                    gt_smplx_vertices[0][None].cpu(), gt_smplx_vertices[1][None].cpu(), squared=False)
                
                # add some metadata 
                thres = 0.1
                results.info['contact_counts'].append((dists < thres).sum().item())
                results.info['img_names'].append(img_name)
                results.info['actions'].append(img_name.split('_')[1].split(' ')[0])
                results.info['subjects'].append(subject)
                
                # PCC (not implemented for BEV because bev estimate is SMPL)
                #pgt_cmap_heat = cmapper.get_full_heatmap(gt_smplx_vertices[0], gt_smplx_vertices[1])
                #max_points = gt_contact_map.sum()
                #distances = pgt_cmap_heat[gt_contact_map]
                #batch_dist = distances[None].repeat(len(results.pcc_x), 1) \
                #    < results.pcc_x.unsqueeze(1).repeat(1, max_points)
                #pcc = batch_dist.sum(1) / max_points
                #results.output['est_pcc'].append(pcc)
                

                onetwo = pairwise_pa_mpjpe_metric(torch.cat(est_smplx_joints, dim=1).cpu().numpy(),torch.cat(gt_smplx_joints, dim=1).cpu().numpy()).mean()
                twoone = pairwise_pa_mpjpe_metric(torch.cat(est_smplx_joints, dim=1).cpu().numpy(),torch.cat([gt_smplx_joints[1], gt_smplx_joints[0]], dim=1).cpu().numpy()).mean()
                #print(cam, onetwo, twoone)
                if twoone < onetwo:
                    gt_smplx_joints = [gt_smplx_joints[1], gt_smplx_joints[0]]
                    gt_smplx_vertices = [gt_smplx_vertices[1], gt_smplx_vertices[0]]
                
                results.output[f'est_mpjpe_h0'].append(
                    mpjpe_metric(est_smplx_joints[0], gt_smplx_joints[0]).mean())
                results.output[f'est_mpjpe_h1'].append(
                    mpjpe_metric(est_smplx_joints[1], gt_smplx_joints[1]).mean())
                results.output[f'est_scale_mpjpe_h0'].append(
                    scale_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), gt_smplx_joints[0].cpu().numpy()).mean())
                results.output[f'est_scale_mpjpe_h1'].append(
                    scale_mpjpe_metric(est_smplx_joints[1].cpu().numpy(), gt_smplx_joints[1].cpu().numpy()).mean())
                results.output[f'est_pa_mpjpe_h0'].append(
                    pa_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), gt_smplx_joints[0].cpu().numpy()).mean())
                results.output[f'est_pa_mpjpe_h1'].append(
                    pa_mpjpe_metric(est_smplx_joints[1].cpu().numpy(), gt_smplx_joints[1].cpu().numpy()).mean())
                results.output[f'est_pa_mpjpe_h0h1'].append(pairwise_pa_mpjpe_metric(
                    torch.cat(est_smplx_joints, dim=1).cpu().numpy(), 
                    torch.cat(gt_smplx_joints, dim=1).cpu().numpy()).mean())


    for metric in ['est_mpjpe_h0', 'est_mpjpe_h1', 'est_pa_mpjpe_h0h1']:
        for action in sorted(set(results.info['actions'])):
            results.get_action_mean(metric, results.info['actions'], action)

    for metric in ['est_mpjpe_h0', 'est_mpjpe_h1', 'est_pa_mpjpe_h0h1']: 
        for subject in sorted(set(results.info['subjects'])):
            results.get_subject_mean(metric, results.info['subjects'], subject)

    results.topkl(print_result=cmd_args.print_result)

    # create a plot with cc on x-axis and est_pa_mpjpe_h0h1 on y-axis
    # thres = 0.1
    # count_thres = [(x < thres).sum().item() for x in results.into['contact_counts']]
    plt.scatter(results.info['contact_counts'], results.output['est_pa_mpjpe_h0h1'])
    # x axis range to 0 - 1000
    plt.xlim([0, 100000])
    plt.ylim([0, 0.5])
    # save plot to file
    plt.savefig(f'{PREDICTIONS_FOLDER}/contact_count_<{thres}_vs_est_pa_mpjpe_h0h1.png')

    # images with highest errors
    if not 'transformer_baseline' in PREDICTIONS_FOLDER:
        img_names = np.array(results.info['img_names'])
        for kk in ['est_pa_mpjpe_h0h1', 'est_mpjpe_h0', 'est_mpjpe_h1']:
            kk_error = np.array(results.output[kk])
            kk_error_idxs = np.argsort(kk_error)
            # worst performers 
            os.makedirs(f'{PREDICTIONS_FOLDER}/worst_performer_{kk}', exist_ok=True)
            kk_sel_idxs = np.concatenate([kk_error_idxs[:10], kk_error_idxs[-10:]])
            for curr_idx in kk_sel_idxs:
                x = img_names[curr_idx]
                y = kk_error[curr_idx] * 1000
                curr_img_path = f'{PREDICTIONS_FOLDER}/images/{x}.png'
                currnew_img_path = f'{PREDICTIONS_FOLDER}/worst_performer_{kk}/{y:00f}__{x}.png'
                shutil.copy(curr_img_path, currnew_img_path)


if __name__ == "__main__":

    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
