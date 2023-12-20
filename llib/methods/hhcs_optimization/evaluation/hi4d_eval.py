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
import math
import smplx 
import shutil
from llib.methods.hhcs_optimization.evaluation.utils import *

from llib.defaults.main import (
    config as default_config,
    merge as merge_configs
)

# We need this to only evaluate on images that do not miss keypoint/BEV detections
ORIG_DATA_FOLDER = f'datasets/original/Hi4D'
PROCESSED_DATA_FOLDER = f'datasets/processed/Hi4D'
PROCESSED_DATA = pickle.load(open(f'{PROCESSED_DATA_FOLDER}/processed.pkl', 'rb'))
TRAIN_VAL_SPLIT = np.load(f'{PROCESSED_DATA_FOLDER}/train_val_test_split.npz')
CAMERAS = ['16', '28', '4', '40', '52', '64', '76', '88']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg', 
        type=str, dest='exp_cfgs', nargs='+', default='llib/methods/hhcs_optimization/evaluation/hi4d_eval.yaml', 
        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
        nargs='*', help='The configuration of the Detector') 
    parser.add_argument('--predictions-folder', type=str, 
        default=f'results/HHC/optimization/fit_heuristic_hi4d_val')
    parser.add_argument('--eval-split', default='val', type=str, choices=['train', 'val', 'test'])
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

    # build metrics 
    #contact_metric = build_metric(cfg.evaluation.cmap_iou)
    scale_mpjpe_metric = build_metric(cfg.evaluation.scale_mpjpe)
    mpjpe_metric = build_metric(cfg.evaluation.mpjpe)
    pa_mpjpe_metric = build_metric(cfg.evaluation.pa_mpjpe)
    pairwise_pa_mpjpe_metric = build_metric(cfg.evaluation.pairwise_pa_mpjpe)

    # SMPL model for predictions
    model_folder = osp.join('essentials/body_models')
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
    bm_smplx_gt = smplx.create(
        model_path=model_folder, 
        model_type='smplx',
        batch_size=2,
        num_betas=16,
    ).to(cfg.device)

    results = ResultLogger(
        method_names=['est']
    )
    results.info = {
        'actions': [],
        'subjects': [],
        'contact_counts': [],
        'img_names': [],
    }
    
    for subject in subjects:
        subject_folder = osp.join(PROCESSED_DATA_FOLDER, f'pair{subject}')

        for action in os.listdir(subject_folder):
            
            orig_subject_folder = osp.join(ORIG_DATA_FOLDER, f'pair{subject}', action)
            processed_subject_folder = osp.join(PROCESSED_DATA_FOLDER, f'pair{subject}', action)
            
            # load SMPL params
            gt_params = PROCESSED_DATA[f'pair{subject}'][action]
            frame_ids = np.arange(0, len(gt_params['betas']), 5)
            params_gt, verts_gt, joints_gt = hi4d_get_smplx_gt(frame_ids, gt_params, bm_smplx_gt)
            verts_gt = torch.from_numpy(verts_gt).to(cfg.device).float()

            for cam in CAMERAS:
                for frame_array_idx, frame_id in enumerate(frame_ids):
                    img_name_raw = [x for x in gt_params['image_data'][frame_id][cam].keys()][0]
                    img_name = f'{subject}_{action}_{cam}_{img_name_raw}0'
                    img_path = osp.join(orig_subject_folder, 'images', cam, img_name_raw+'.jpg')
                    PRED_ITEM_PATH = f'{PREDICTIONS_FOLDER}/results/{img_name}.pkl'
                    
                    has_bev = gt_params['image_data'][frame_id][cam][img_name_raw]
                    if (has_bev[0]['bev_human_idx'] == -1 or has_bev[1]['bev_human_idx'] == -1):
                        # print('BEV MISSING', subject, action, frame_id, cam)
                        continue

                    if (has_bev[0]['vitpose_human_idx'] == -1 and has_bev[0]['openpose_human_idx'] == -1) or \
                        (has_bev[1]['vitpose_human_idx'] == -1 and has_bev[1]['openpose_human_idx'] == -1):
                        # print('KPT MISSING', subject, action, frame_id, cam)
                        continue
                    
                    # check if item was in processing / optimization batch or if information was missing
                    if not os.path.exists(PRED_ITEM_PATH):
                        print('PREDICTION MISSING', subject, action, frame_id, cam)
                        continue

                    # predicted data
                    pred_item = pickle.load(open(PRED_ITEM_PATH, 'rb'))
                    params_pred, verts_pred, joints_pred = get_smplx_pred(pred_item['humans'], bm_smplxa)

                    if False: #True:
                        for iii in [0,1]:
                            save_mesh(verts_gt[iii, frame_array_idx].cpu().numpy(), bm_smplx_gt.faces, f'outdebug/hi4d_joint_pa_mpjpe/hi4d_{subject}_{action}_{cam}_test_mesh_gt_{iii}.ply')
                            save_mesh(verts_pred[iii].cpu().numpy(), bm_smplx_gt.faces, f'outdebug/hi4d_joint_pa_mpjpe/hi4d_{subject}_{action}_{cam}_test_mesh_pred_{iii}.ply')
                    
                    est_smplx_vertices = [verts_pred[0], verts_pred[1]]
                    est_smplx_joints = [verts2joints(verts_pred[0][None], J14_REGRESSOR), verts2joints(verts_pred[1][None], J14_REGRESSOR)]
                    gt_smplx_vertices = [verts_gt[0, frame_array_idx], verts_gt[1, frame_array_idx]]
                    gt_smplx_joints = [verts2joints(verts_gt[0, frame_array_idx][None], J14_REGRESSOR), verts2joints(verts_gt[1, frame_array_idx][None], J14_REGRESSOR)]

                    # thres = 0.1
                    # dists = pcl_pcl_pairwise_distance(
                        # est_smplx_vertices[0][None].cpu(), est_smplx_vertices[1][None].cpu(), squared=False)
                    
                    #import ipdb;ipdb.set_trace()
                    onetwo = pairwise_pa_mpjpe_metric(torch.cat(est_smplx_joints, dim=1).cpu().numpy(),torch.cat(gt_smplx_joints, dim=1).cpu().numpy()).mean()                                      
                    twoone = pairwise_pa_mpjpe_metric(torch.cat(est_smplx_joints, dim=1).cpu().numpy(),torch.cat([gt_smplx_joints[1], gt_smplx_joints[0]], dim=1).cpu().numpy()).mean()
                    #print(cam, onetwo, twoone)
                    if twoone < onetwo:
                        gt_smplx_joints = [gt_smplx_joints[1], gt_smplx_joints[0]]
                        gt_smplx_vertices = [gt_smplx_vertices[1], gt_smplx_vertices[0]]
                        #import ipdb;ipdb.set_trace()
                        #for iii in [0,1]:
                        #    save_mesh(verts_gt[iii, frame_array_idx].cpu().numpy(), bm_smplx_gt.faces, f'outdebug/hi4d_joint_pa_mpjpe/hi4d_{subject}_{action}_{cam}_test_mesh_gt_{iii}.ply')
                        #    save_mesh(verts_pred[iii].cpu().numpy(), bm_smplx_gt.faces, f'outdebug/hi4d_joint_pa_mpjpe/hi4d_{subject}_{action}_{cam}_test_mesh_pred_{iii}.ply')
                    # results.info['contact_counts'].append((dists < thres).sum().item())
                    results.info['img_names'].append(img_name)
                    results.info['actions'].append(action[:-2])
                    results.info['subjects'].append(subject)
               
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
