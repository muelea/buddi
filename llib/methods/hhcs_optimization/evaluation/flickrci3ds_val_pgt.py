import argparse
import numpy as np
import cv2
import torch 
import os
import json
from tqdm import tqdm
import os.path as osp
from loguru import logger as guru
import pickle
import datetime
import matplotlib.pyplot as plt
from llib.visualization.utils import *
from llib.utils.metrics.build import build_metric
from llib.bodymodels.build import build_bodymodel
from llib.utils.threed.distance import ContactMap
import smplx
from llib.defaults.main import (
    config as default_config,
    merge as merge_configs
)

PROJECT_HOME = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact' #os.environ['HUMANHUMANCONTACT_HOME']
ESSENTIALS_HOME = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact/essentials/' #os.environ['ESSENTALS_HOME']
'''
JOINT_SUBSET = []
SUBSET_PATH = f'{PROJECT_HOME}/datasets/processed/FlickrCI3D_Signatures_Transformer/joint_subset_flickrci3ds_test.npy'
if osp.exists(SUBSET_PATH):
    JOINT_SUBSET = np.load(SUBSET_PATH).tolist()
else:
    M1 = os.listdir('/is/cluster/work/lmueller2/results/HHC/optimization/fit_pseudogt_flickrci3ds_test/results')
    M2 = os.listdir('/is/cluster/work/lmueller2/results/HHC/optimization/fit_baseline_flickrci3ds_test/results')
    M3 = os.listdir('/is/cluster/work/lmueller2/results/HHC/optimization/fit_diffprior_flickrci3ds_test/results')
    for m in M1:
        if not m.endswith('.pkl'):
            continue
        if m in M2 and m in M3:
            JOINT_SUBSET.append(m.split('.')[0])
    # save joint subset to numpy file
    np.save(SUBSET_PATH, JOINT_SUBSET)
print(f'NUMBER OF SAMPLES IN TEST SET: {len(JOINT_SUBSET)}')
'''
# SMPL JOINT REGRESSOR
REGION_TO_VERTEX_PATH = osp.join(ESSENTIALS_HOME, 'contact/flickrci3ds_r75_rid_to_smplx_vid.pkl')
J14_REGRESSOR_PATH = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/SMPLX_to_J14.pkl'
J14_REGRESSOR = torch.from_numpy(pickle.load(open(J14_REGRESSOR_PATH, 'rb'), encoding='latin1')).to('cuda').float()
# Indices to get the 14 LSP joints from the ground truth SMPL joints
jreg_path = f'{ESSENTIALS_HOME}/body_model_utils/joint_regressors/J_regressor_h36m.npy'
SMPL_TO_H36M = torch.from_numpy(np.load(jreg_path)).to('cuda').float()
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


OURS_OPTI_FOLDER = f'{PROJECT_HOME}/outdebug/downstream_optimization/OURS_hhcloss_refined_noscale_v3'
OURS_FOLDER = f'{PROJECT_HOME}/results/transformer/OURS_FLICKRCI3D_SigVal/match_val_pkl/results'
PGT_FILE = f'{PROJECT_HOME}/datasets/processed/FlickrCI3D_Signatures_Transformer/val.pkl'
EST_FILE = f'/shared/lmueller/projects/humanhumancontact/results/transformer/OURS_FLICKRCI3D_SigVal/match_val_pkl/results/OURS_estimates.pkl'

PSGT_FOLDER = '/is/cluster/work/lmueller2/results/HHC/optimization/fit_pseudogt_flickrci3ds_val'
BASELINE_FOLDER = '/is/cluster/work/lmueller2/results/HHC/optimization/fit_baseline_flickrci3ds_val'
DATASET_FOLDER = f'{PROJECT_HOME}/datasets/original/FlickrCI3D_Signatures'

DATA_PROCESSED_TEST = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact/datasets/processed/FlickrCI3D_Signatures_Transformer/test.pkl'
DATA_PROCESSED_VAL = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact/datasets/processed/FlickrCI3D_Signatures_Transformer/val.pkl'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg',  type=str, 
        dest='exp_cfgs', nargs='+', default='eval_flickrci3ds.yaml', 
        help='The configuration of the experiment')
        
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
        nargs='*', help='The configuration of the Detector') 

    parser.add_argument('--flickrci3ds-folder', type=str, default=DATASET_FOLDER,
        help='Folder where the flickrci3ds dataset is stored.')

    parser.add_argument('--flickrci3ds-split', type=str, default='val',
        help='The dataset split to evaluate on')

    parser.add_argument('-gt', '--pseudo-ground-truth', type=str, default=PSGT_FOLDER,
        help='Folder where the pseudo ground-truth meshes are stored.')

    parser.add_argument('-p','--predicted', type=str, default=BASELINE_FOLDER,
        help='Folder where the predicted meshes / method to evaluate against are \
            stored. To evaluate BEV, pass the processed pkl file.')

    parser.add_argument('--predicted-is-bev', default=False, action='store_true',
        help='Set flag if predicted meshes are bev estimates (because they are SMPL).')

    parser.add_argument('--predicted-output-folder', type=str, default='temp',
        help='Folder where the optimized meshes are stored.')
    
    parser.add_argument('--use-joint-subset', type=str, default=False,
        help='Use a subset of the Flickr data for which all methods have estimates.')
    
    cmd_args = parser.parse_args()

    cfg = merge_configs(cmd_args, default_config)

    return cfg, cmd_args

class ResultLogger():
    def __init__(
        self, 
        method_names=[]
    ):

        self.method_names = method_names
        self.metric_names = [
            'mpjpe_h0',
            'mpjpe_h1',
            'scale_mpjpe_h0',
            'scale_mpjpe_h1',
            'pa_mpjpe_h0',
            'pa_mpjpe_h1',
            'pa_mpjpe_h0h1',
            'cmap_heat',
            'iou',
            'precision',
            'recall',
            'fscore',
            'pcc'

        ]
        
        self.pcc_x = torch.from_numpy(np.arange(0.0, 1.0, 0.05)).to('cuda')

        self.init_result_dict()

    def init_result_dict(self):

        self.output = {}

        for mm in self.method_names:
            for metric in self.metric_names:
                self.output[f'{mm}_{metric}'] = []

def get_smplx_params(human, body_model_smplx=None):
    """
    Returns the SMPL parameters of a human.
    """
    def to_tensor(x):
        return torch.tensor(x).to('cuda')

    params = dict(
        betas = to_tensor(human[f'betas']),
        global_orient = to_tensor(human[f'global_orient']),
        body_pose = to_tensor(human[f'body_pose']),
        scale = to_tensor(human[f'scale']),
        translx = to_tensor(human[f'translx']),
        transly = to_tensor(human[f'transly']),
        translz = to_tensor(human[f'translz']),
    )

    verts, joints = None, None
    if body_model_smplx is not None:
        body = body_model_smplx(**params)
        verts = body.vertices.detach() 
        joints = torch.matmul(J14_REGRESSOR, verts)

    return params, verts, joints

    # load bev data 
def get_bev_smplx_params(item, human_idx):
    """Read SMPL parameters from a pkl file."""

    bev_smplx_vertices = torch.from_numpy(item[f'vertices_h{human_idx}']).unsqueeze(0).to('cuda')
    #bev_smplx_joints = torch.matmul(J14_REGRESSOR, bev_smplx_vertices)

    bev_smpl_vertices = torch.from_numpy(item[f'bev_orig_vertices_h{human_idx}']).unsqueeze(0).to('cuda')
    bev_joints = torch.matmul(SMPL_TO_H36M, bev_smpl_vertices)[:,H36M_TO_J14,:]

    return bev_smplx_vertices, bev_joints


def main(cfg, cmd_args):
    """
    Evaluates the performance of a method on the FlickrCI3D dataset.
    Compares the method to pseudo-ground-truth.
    """

    # build metrics 
    contact_metric = build_metric(cfg.evaluation.cmap_iou)
    scale_mpjpe_metric = build_metric(cfg.evaluation.scale_mpjpe)
    mpjpe_metric = build_metric(cfg.evaluation.mpjpe)
    pa_mpjpe_metric = build_metric(cfg.evaluation.pa_mpjpe)
    pairwise_pa_mpjpe_metric = build_metric(cfg.evaluation.pairwise_pa_mpjpe)

    # load ground-truth data from FlickrCI3D folder
    if cmd_args.flickrci3ds_split == 'val':
        dataset_folder = f'{cmd_args.flickrci3ds_folder}/train'
        train_val_split = np.load(f'{dataset_folder}/train_val_split.npz')
        img_names = train_val_split['val']
    elif cmd_args.flickrci3ds_split == 'test':
        dataset_folder = f'{cmd_args.flickrci3ds_folder}/test'
        img_names = os.listdir(f'{dataset_folder}/images')
        img_names = [x.split('.')[0] for x in img_names]
    
    if cmd_args.predicted_is_bev:
        est_data = pickle.load(open(cmd_args.predicted, 'rb'))
        bev_data = {}
        for item_idx, item in enumerate(est_data):
            imgname = item['imgname'].split('.')[0]
            if imgname in bev_data.keys():
                bev_data[imgname][item['contact_index']] = item_idx
            else:
                bev_data[imgname] = {item['contact_index']: item_idx}
    else:
        DATA_PROCESSED = DATA_PROCESSED_TEST if cmd_args.flickrci3ds_split == 'test' else DATA_PROCESSED_VAL
        bev_data_orig = pickle.load(open(DATA_PROCESSED, 'rb'))
        bev_data = {}
        for item_idx, item in enumerate(bev_data_orig):
            imgname = item['imgname'].split('.')[0]
            if imgname in bev_data.keys():
                bev_data[imgname][item['contact_index']] = item_idx
            else:
                bev_data[imgname] = {item['contact_index']: item_idx}


    # contact map annotations
    annotations_path = f'{dataset_folder}/interaction_contact_signature.json'
    annotations = json.load(open(annotations_path, 'r'))

    pseudo_gt_folder = cmd_args.pseudo_ground_truth

    # load body model to get joints from SMPL parameters
    body_model_smplx = build_bodymodel(
        cfg=cfg.body_model, 
        batch_size=cfg.batch_size, 
        device=cfg.device
    )

    cmapper = ContactMap(
        region_to_vertex=REGION_TO_VERTEX_PATH,
    )

    results = ResultLogger(
        method_names=['est']
    )

    contact_zeros = torch.zeros((75, 75)) \
        .to(torch.bool).unsqueeze(0).to('cuda')

    #for item_idx, item in tqdm(enumerate(pgt_data)):
    total_size, pgt_not_found, est_not_found = 0, 0, 0
    for item_idx, item in tqdm(enumerate(img_names)):
        if item not in img_names:
            continue
        annos = annotations[item]

        for anno_idx, anno in enumerate(annos['ci_sign']):
            total_size += 1
            p1_idx, p2_idx = anno['person_ids']
            region_id = anno['smplx']['region_id']

            if cmd_args.use_joint_subset:
                if f'{item}_{anno_idx}' not in JOINT_SUBSET:
                    continue

            gt_cmap = contact_zeros.clone()
            for rid in region_id:
                gt_cmap[0, rid[0], rid[1]] = True

            # load pseudo ground-truth
            pgt_smplx_vertices, pgt_smplx_joints = [], []
            pgt_filename = osp.join(cmd_args.pseudo_ground_truth, 'results', f'{item}_{anno_idx}.pkl')
            if not osp.exists(pgt_filename):
                pgt_not_found += 1
                #print(f'Pseudo ground-truth not found for {item}_{anno_idx}')
                continue
            pgt_item = pickle.load(open(pgt_filename, 'rb'))
            for hidx in range(2):
                _, verts, joints = get_smplx_params(
                    pgt_item[f'h{hidx}'], body_model_smplx)
                pgt_smplx_vertices.append(verts)
                pgt_smplx_joints.append(joints)

            # load estimated
            est_smplx_vertices, est_smplx_joints = [], []
            if cmd_args.predicted_is_bev:
                est_item_idx = bev_data[item][anno_idx]
                est_item = est_data[est_item_idx]
                # flip h0 h1 if needed
                h0id = 0 if est_item['transl_h0'][0] <= est_item['transl_h1'][0] else 1
                h1id = 1-h0id
                bev_hids = [h0id, h1id]
                if h0id == 1:
                    gt_cmap = torch.transpose(gt_cmap, 1, 2)
            else:
                est_filename = osp.join(cmd_args.predicted, 'results', f'{item}_{anno_idx}.pkl')
                if not osp.exists(est_filename):
                    est_not_found += 1
                    print(f'Estimate not found for {item}_{anno_idx}.')
                    continue
                est_item = pickle.load(open(est_filename, 'rb'))
                # contact map if p1, p2 were flipped 
                bev_item_idx = bev_data[item][anno_idx]
                bev_item = bev_data_orig[bev_item_idx]
                # flip h0 h1 if needed
                h0id = 0 if bev_item['transl_h0'][0] <= bev_item['transl_h1'][0] else 1
                if h0id == 1:
                    gt_cmap = torch.transpose(gt_cmap, 1, 2)
            for hidx in range(2):
                if cmd_args.predicted_is_bev:
                    verts, joints = get_bev_smplx_params(est_item, bev_hids[hidx])
                else:
                    _, verts, joints = get_smplx_params(
                        est_item[f'h{hidx}'], body_model_smplx)
                est_smplx_vertices.append(verts)
                est_smplx_joints.append(joints)

            # PGT to GT Contact Map error (as in 3D human interactions)
            pgt_cmap_heat = cmapper.get_full_heatmap(est_smplx_vertices[0], est_smplx_vertices[1])
            pgt_cmap_binary = pgt_cmap_heat < 0.013
            iou, precision, recall, fsore = contact_metric(pgt_cmap_binary, gt_cmap)
            results.output['est_iou'].append(iou)
            results.output['est_precision'].append(precision)
            results.output['est_recall'].append(recall)
            results.output['est_fscore'].append(fsore)

            # PCC
            max_points = gt_cmap.sum()
            distances = pgt_cmap_heat[gt_cmap]
            batch_dist = distances[None].repeat(len(results.pcc_x), 1) \
                < results.pcc_x.unsqueeze(1).repeat(1, max_points)
            pcc = batch_dist.sum(1) / max_points
            results.output['est_pcc'].append(pcc)

            # MPJPE, PA-MPJPE, PA-MPJPE (pairwise) between our and GT
            def to_j14(x, J14_REGRESSOR=J14_REGRESSOR):
                return torch.matmul(J14_REGRESSOR, x)
            est_smplx_joints = [to_j14(est_smplx_vertices[0]), to_j14(est_smplx_vertices[1])]
            pgt_smplx_joints = [to_j14(pgt_smplx_vertices[0]), to_j14(pgt_smplx_vertices[1])]

            results.output[f'est_mpjpe_h0'].append(
                mpjpe_metric(est_smplx_joints[0], pgt_smplx_joints[0]).mean())
            results.output[f'est_mpjpe_h1'].append(
                mpjpe_metric(est_smplx_joints[1], pgt_smplx_joints[1]).mean())
            results.output[f'est_scale_mpjpe_h0'].append(
                scale_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), pgt_smplx_joints[0].cpu().numpy()).mean())
            results.output[f'est_scale_mpjpe_h1'].append(
                scale_mpjpe_metric(est_smplx_joints[1].cpu().numpy(), pgt_smplx_joints[1].cpu().numpy()).mean())
            results.output[f'est_pa_mpjpe_h0'].append(
                pa_mpjpe_metric(est_smplx_joints[0].cpu().numpy(), pgt_smplx_joints[0].cpu().numpy()).mean())
            results.output[f'est_pa_mpjpe_h1'].append(
                pa_mpjpe_metric(est_smplx_joints[1].cpu().numpy(), pgt_smplx_joints[1].cpu().numpy()).mean())
            results.output[f'est_pa_mpjpe_h0h1'].append(pairwise_pa_mpjpe_metric(
                torch.cat(est_smplx_joints, dim=1).cpu().numpy(), torch.cat(pgt_smplx_joints, dim=1).cpu().numpy()).mean())
    
    if cmd_args.predicted_is_bev:
        output_folder = cmd_args.predicted_output_folder
    else:
        output_folder = cmd_args.predicted
    evaluation_folder = osp.join(output_folder, 'evaluation')
    os.makedirs(evaluation_folder, exist_ok=True)
    result_fn = os.path.join(evaluation_folder, f'metrics.txt')
    output_file = open(result_fn, 'a')
    # write date and time in readable format to output file
    _ = output_file.write(str(datetime.datetime.now()))

    for k, v in results.output.items():
        if len(v) == 0:
            continue

        if isinstance(v[0], torch.Tensor):
            v = torch.stack(v).cpu().numpy()
        else:
            v = np.array(v)

        if 'mpjpe' in k: # convert to mm
            v = v * 1000
        
        if 'pcc' in k:
            pcc = v.mean(0)
            # plot with x axis from result.pcc_X as pcc as value
            plt.plot(results.pcc_x.cpu().numpy(), pcc)
            plt.xlabel('Distance threshold (m)')
            plt.ylabel('PCC')
            plt.savefig(os.path.join(evaluation_folder, f'pcc.png'))
            plt.close()
            pcc_str = ';'.join([str(x) for x in pcc])
            print(pcc_str)
            x_str = ';'.join([str(x) for x in results.pcc_x.cpu().numpy()])
            _ = output_file.write(f'{k}_x: {x_str}')
            _ = output_file.write(f'{k}: {pcc_str}')


        error = v.mean()
        print(f'{k}: {error}')
        # write to file in predicted folder
        _ = output_file.write(f'{k}: {error} \n')
    
    _ = output_file.write(f'Pseudo ground-truth not found: {pgt_not_found}/{total_size}')
    _ = output_file.write(f'Predicted not exsits: {est_not_found}/{total_size}')
    output_file.close()

if __name__ == "__main__":

    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
