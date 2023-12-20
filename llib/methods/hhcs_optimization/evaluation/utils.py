import numpy as np
import torch
import json
import os
from tqdm import tqdm 
import os.path as osp
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms import euler_angles_to_matrix
import trimesh
import pickle

# ESSENTIALS_HOME = 'essentials' #os.environ['ESSENTIALS_HOME']
# PROJECT_HOME = '/is/cluster/lmueller2/projects/HumanHumanContact/humanhumancontact' #os.environ['HUMANHUMANCONTACT_HOME']
REGION_TO_VERTEX_PATH = 'essentials/contact/flickrci3ds_r75_rid_to_smplx_vid.pkl'

J14_REGRESSOR_PATH = f'essentials/body_model_utils/joint_regressors/SMPLX_to_J14.pkl'
J14_REGRESSOR = torch.from_numpy(
    pickle.load(open(J14_REGRESSOR_PATH, 'rb'), encoding='latin1')).to('cuda').float()

# Indices to get the 14 LSP joints from the ground truth SMPL joints
jreg_path = 'essentials/body_model_utils/joint_regressors/J_regressor_h36m.npy'
SMPL_TO_H36M = torch.from_numpy(np.load(jreg_path)).to('cuda').float()
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

def chi3d_items_for_eval(subjects, split_folder, orig_data_folder, processed_data):

    out = {}
    for subject in tqdm(subjects):
        annotation_fn =  osp.join(
            orig_data_folder, split_folder, subject, 'interaction_contact_signature.json')
        annotations = json.load(open(annotation_fn, 'r'))

        for action, annotation in tqdm(annotations.items()):
            orig_subject_folder = osp.join(orig_data_folder, split_folder, subject)
            frame_id = annotation['fr_id']
            cameras = os.listdir(osp.join(orig_subject_folder, 'camera_parameters'))

            for cam in cameras:
                if f'{action}_{frame_id:06d}_{cam}' not in processed_data[subject].keys():
                    continue
                
                # check if item was in processing / optimization batch or if information was missing
                missing_info = False
                for cidx in [0,1]:
                    for dd in ['openpose_human_idx', 'bev_human_idx', 'vitpose_human_idx', 'vitposeplus_human_idx']:
                        if processed_data[subject][f'{action}_{frame_id:06d}_{cam}'][cidx][dd] == -1:
                            missing_info = True
                if missing_info:
                    continue

                if subject not in out.keys():
                    out[subject] = {}
                if action not in out[subject].keys():
                    out[subject][action] = []

                out[subject][action] += [cam]

    return out

def chi3d_get_smplx_gt(smpl_fn, frame_ids, body_model):

    smplx_data = json.load(open(smpl_fn, 'r'))
    for k, v in smplx_data.items():
        smplx_data[k] = np.array(v)

    num_frames = len(frame_ids)

    params = {
        'global_orient': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['global_orient'][:,frame_ids,:]))).float()[:,:,0,:],
        'transl': torch.from_numpy(np.array(smplx_data['transl'][:,frame_ids,:])).float(),
        'body_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['body_pose'][:,frame_ids,:]))).view(2, num_frames, -1).float(), 
        'betas': torch.from_numpy(np.array(smplx_data['betas'][:,frame_ids,:])).float(),
        'left_hand_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['left_hand_pose'][:,frame_ids,:]))).float().view(2, num_frames, -1)[:,:,:6],
        'right_hand_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['right_hand_pose'][:,frame_ids,:]))).float().view(2, num_frames, -1)[:,:,:6], 
        'jaw_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['jaw_pose'][:,frame_ids,:]))).float()[:,:,0,:], 
        'leye_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['leye_pose'][:,frame_ids,:]))).float()[:,:,0,:], 
        'reye_pose': matrix_to_axis_angle(torch.from_numpy(np.array(smplx_data['reye_pose'][:,frame_ids,:]))).float()[:,:,0,:], 
        'expression': torch.from_numpy(np.array(smplx_data['expression'][:,frame_ids,:])).float(),
    }

    vertices = np.zeros((2, num_frames, 10475, 3)) #body.vertices.detach().to('cpu').numpy()[0]
    joints = np.zeros((2, num_frames, 127, 3)) #body.joints.detach().to('cpu').numpy()[0]

    for array_idx in range(num_frames):
        params_for_smpl = {} 
        for key, val in params.items():
            params_for_smpl[key] = val[:, array_idx, :].to('cuda')
        body = body_model(**params_for_smpl)
        vertices[:, array_idx] = body.vertices.detach().to('cpu').numpy()
        joints[:, array_idx] = body.joints.detach().to('cpu').numpy()
    
    return params, vertices, joints

def chi3d_read_cam_params(cam_path):
    with open(cam_path) as f:
        cam_params = json.load(f)
        for key1 in cam_params:
            for key2 in cam_params[key1]:
                cam_params[key1][key2] = np.array(cam_params[key1][key2]) 
    return cam_params

def chi3d_verts_world2cam(verts_world, cam_params):
    TT_GT = torch.tensor(cam_params['extrinsics']['T']).to("cuda")
    RR_GT = torch.tensor(cam_params['extrinsics']['R']).to("cuda")
    verts_camera = torch.matmul(verts_world[:,0,:,:] - TT_GT, torch.transpose(RR_GT, 1, 0)).float()
    return verts_camera

def chi3d_bev_verts_from_processed_data(processed_data, subject, action, cam, frame_id, device='cuda'):
    img_name = f'{action}_{frame_id:06d}_{cam}'
    bev_vertices_0 = processed_data[subject][img_name][0]['bev_smpl_vertices'] 
    bev_vertices_1 = processed_data[subject][img_name][1]['bev_smpl_vertices']
    bev_transl_0 = processed_data[subject][img_name][0]['bev_cam_trans'] 
    bev_transl_1 = processed_data[subject][img_name][1]['bev_cam_trans']
    bev_vertices = torch.from_numpy(np.concatenate([bev_vertices_0[None], bev_vertices_1[None]], axis=0)).to(device)
    bev_transl = torch.from_numpy(np.concatenate([bev_transl_0[None][None], bev_transl_1[None][None]], axis=0)).to(device)
    bev_vertices = bev_vertices + bev_transl
    return bev_vertices


def verts2joints(x, joint_regressor):
    return torch.matmul(joint_regressor, x)

def save_mesh(verts, faces, fn='outdebug/test_mesh.ply'):
    mesh = trimesh.Trimesh(verts, faces)
    _ = mesh.export(fn)

def hi4d_get_smplx_gt(frame_ids, gt_params, body_model):


    num_frames = len(frame_ids)

    params = {
        'global_orient': torch.from_numpy(gt_params['smplx_global_orient_unit'][frame_ids]),
        'transl': torch.from_numpy(gt_params['smplx_transl_unit'][frame_ids]),
        'body_pose': torch.from_numpy(gt_params['smplx_body_pose'][frame_ids]).view(num_frames, 2, -1),
        'betas': torch.from_numpy(gt_params['smplx_betas'][frame_ids])[:,:,:10],
    }

    vertices = np.zeros((2, num_frames, 10475, 3)) #body.vertices.detach().to('cpu').numpy()[0]
    joints = np.zeros((2, num_frames, 127, 3)) #body.joints.detach().to('cpu').numpy()[0]

    for array_idx in range(num_frames):
        params_for_smpl = {} 
        for key, val in params.items():
            params_for_smpl[key] = val[array_idx, :, :].to('cuda')
        body = body_model(**params_for_smpl)
        vertices[:, array_idx] = body.vertices.detach().to('cpu').numpy()
        joints[:, array_idx] = body.joints.detach().to('cpu').numpy()
    
    return params, vertices, joints


def tabilize(results, header_label, precisions, rank_order, suffixes=None, hlines = [], textbf=False, print_console=True):

    # Example:
    # results = {
    #     'That one algorithm': [30.52436, 0.151243, 9.61],
    #     'That other algorithm': [32.1315, 0.074125, 100.1231],
    #     'Yet another algorithm': [19.26456, 0.43312, 3.10],
    #     'My beloved algorithm': [38.924123, 0.051241, 60.3145]}
    # precisions = [2, 3, 0]  # How many digits of precision to use.
    # rank_order = [1, -1, 0]  # +1 = higher is better, -1 = lower is better, 0 = do not color code.
    # suffixes = ['', '', ' sec.']  # What string to append after each number.
    # hlines = [3] # Where to insert horizontal lines.
    # tabilize(results, precisions, rank_order, suffixes=suffixes, hlines=hlines)


    def rankify(x, order):
        # Turn a vector of values into a list of ranks, while handling ties.
        assert len(x.shape) == 1
        if order == 0:
            return np.full_like(x, 1e5, dtype=np.int32)
        u = np.sort(np.unique(x))
        if order == 1:
            u = u[::-1]
        r = np.zeros_like(x, dtype=np.int32)
        for ui, uu in enumerate(u):
            mask = x == uu
            r[mask] = ui
        return np.int32(r)

    def to_console_readable(line):
        line = line.replace('\cellcolor{tabfirst}', '').replace('\cellcolor{tabsecond}', '') \
                .replace('\cellcolor{tabthird}', '').replace('\\textbf{{', '') \
                .replace('}', '').replace('\\', '')
        line = line.split('&')
        row_name = line[0]
        row_vals = line[1:]
        # pad with spaces to size of longest line
        max_length = 10
        row_vals = [f"{x.strip():<{max_length}}" for x in row_vals]  # pad with spaces to max_length
        line = ' | '.join([row_name] + row_vals)
        return line
    
    names = results.keys()
    data = np.array(list(results.values()))
    assert len(names) == len(data)
    data = np.array(data)

    if not textbf:
        tags = [' \cellcolor{tabfirst}',
                '\cellcolor{tabsecond}',
                ' \cellcolor{tabthird}',
                '                     ']
    else:
        tags = [' \\textbf{{            ',
                '                     ',
                '                     ']
    
    max_len = max([len(v) for v in list(names)])
    names_padded = [v + ' '*(max_len-len(v)) for v in names]

    data_quant = data.copy()
    # round each item in data_quant if it is not a string
    for col in range(data_quant.shape[1]):
        for row in range(data_quant.shape[0]):
            if not isinstance(data_quant[row,col], str):
                data_quant[row,col] = np.round(data_quant[row,col], precisions[col])


    #data_quant = np.round((data * 10.**(np.array(precisions)[None, :]))) / 10.**(np.array(precisions)[None, :])
    if suffixes is None:
        suffixes = [''] * len(precisions)

    tagranks = []
    for d in range(data_quant.shape[1]):
        tagranks.append(np.clip(rankify(data_quant[:,d], rank_order[d]), 0, len(tags)-1))
    tagranks = np.stack(tagranks, -1)

    # create header / column names 
    cell_len = max_len
    header = ' ' * (max_len) + ' & '
    for x_idx, x in enumerate(header_label):
        header += f'{x}' + ' ' * (cell_len-len(x)) 
        if not x_idx == len(header_label)-1:
            header += '& '
    print(header)

    # create table content
    for i_row in range(len(names)):
        line = ''
        if i_row in hlines:
            line += '\\hline\n'
        line += names_padded[i_row]
        for d in range(data_quant.shape[1]):
            cell = ' & '
            if rank_order[d] != 0 and not np.isnan(data[i_row,d]):
                cell += tags[tagranks[i_row, d]]
            if np.isnan(data[i_row,d]):
                cell += ' - '
            else:
                assert precisions[d] >= 0
                if data_quant[i_row,d] == 'NaN':
                    cell_precision = ''
                else:
                    cell_precision = f'0.{precisions[d]}f'
                cell += ('{:' + cell_precision + '}').format(data_quant[i_row,d]) + suffixes[d]
                # if bold text add closing bracket
                if textbf and tagranks[i_row, d] == 0:
                    cell += '}'
            # check length of cell
            line += cell + ' ' * (cell_len-len(cell)-1)
        if i_row < len(names)-1:
            line += ' \\\\'

        if print_console:
            line = to_console_readable(line)
            
        print(line)



class ResultLogger():
    def __init__(
        self, 
        method_names=[],
        output_fn=None,
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

        self.info = {}

        self.pkl = {}
        
        self.pcc_x = torch.from_numpy(np.arange(0.0, 1.0, 0.05)).to('cuda')

        self.output_fn = output_fn
        self.temp_output_fn = 'llib/methods/hhcs_optimization/evaluation/temp/result.pkl'

        self.init_result_dict()

    def init_result_dict(self):

        self.output = {}

        for mm in self.method_names:
            for metric in self.metric_names:
                self.output[f'{mm}_{metric}'] = []

    def get_action_mean(self, metric, label, action):
        # def get_action_mean(value, label, action): 
        value = self.output[metric]
        sumvals = []
        for x, y in zip(label, value):
            if x == action:
                sumvals.append(y)
        error = np.array(sumvals).mean()
        self.output[f'{action}_{metric}'] = [error]
        # print(f'{action}: {error:.1f}')

    def get_subject_mean(self, metric, label, subject):
        value = self.output[metric]
        sumvals = []
        for x, y in zip(label, value):
            if x == subject:
                sumvals.append(y)
        error = np.array(sumvals).mean()
        self.output[f'{subject}_{metric}'] = [error]
        # print(f'{subject}: {error:.1f}')


    def topkl(self, save_pkl=True, print_result=False, to_mm=True):
        """
        Create dict where each key is a error metric.
        """
        for k, v in self.output.items():
            if len(v) == 0:
                continue

            if isinstance(v[0], torch.Tensor):
                v = torch.stack(v).cpu().numpy()
            else:
                v = np.array(v)

            if 'mpjpe' in k and to_mm: # convert to mm
                v = v * 1000
            
            if 'pcc' in k:
                for x, y in zip(self.pcc_x, v.mean(0)):
                    self.pkl[f'{k}_{x:.2f}'] = y * 100
                    if print_result:
                        print(f'{k}_{x:.2f}: {y}')
            else:
                error = v.mean()
                self.pkl[k] = v.mean()
                if print_result:
                    print(f'{k}: {error:.1f}')

        with open(self.temp_output_fn, 'wb') as f:
            pickle.dump(self.pkl, f) 

        if self.output_fn is not None:

            # import wandb
            # import pandas as pd 
            # run = wandb.init(
            #     project = 'HumanHumanContact',
            #     config = {'name': 'Test Evaluation'}
            # )

            # if True:
            #     keys_output = [x for x in self.output.keys() if len(self.output[x]) > 0]
            #     keys_info = [x for x in self.info.keys() if len(self.info[x]) > 0]
            #     data = {}
            #     for kk in keys_output:
            #         if len (self.output[kk]) == 431:
            #             data[kk] = self.output[kk]
            #     for kk in keys_info:
            #         if len(self.info[kk]) == 431:
            #             data[kk] = self.info[kk]
            #     df = pd.DataFrame(data)
            #     tbl = wandb.Table(dataframe=df)
            #     run.log({'results': tbl})


            output_all = {
                'output': self.output,
                'pkl': self.pkl,
                'info': self.info,
                'pcc_x': self.pcc_x
            }

            with open(self.output_fn, 'wb') as f:
                pickle.dump(output_all, f)




def tabilize_wrapper(
    result,
    header_label = ['est_pa_mpjpe_h0', 'est_pa_mpjpe_h1', 'est_mpjpe_h0', 'est_mpjpe_h1', 'est_pa_mpjpe_h0h1'],
    row_label = ['pgt', 'bev', 'heuristic', 'buddi'],
    precisions = -1,
    rank_order = 0,
    suffixes = '',
    textbf=False,
    print_console=False
):
    """
    Takes a dict of dicts, where each first key is a method name and each second key the metric name
    and converts this to dict of lists where each element of list becomes one column in table
    """
    num_cols = len(header_label)

    if isinstance(precisions, int):
        precisions = num_cols * [precisions]

    if isinstance(rank_order, int):
        rank_order = num_cols * [rank_order]

    if isinstance(suffixes, str):
        suffixes = num_cols * [suffixes]

    results_reshaped = {}
    for mm in row_label:
        results_reshaped[mm] = []
        for hh in header_label:
            if hh not in result[mm].keys():
                results_reshaped[mm] += [np.nan]
            else:
                results_reshaped[mm] += [result[mm][hh]]
    
    tabilize(results_reshaped, 
             header_label, precisions, 
             rank_order, suffixes, 
             textbf=textbf, 
             print_console=print_console)