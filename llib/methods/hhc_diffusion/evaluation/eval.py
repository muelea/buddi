from llib.utils.metrics.diffusion import GenFID, GenDiversity
from llib.utils.threed.conversion import axis_angle_to_rotation6d, rotation6d_to_axis_angle
from llib.bodymodels.smpla import SMPLXA
from torch.utils.data import DataLoader
from llib.data.build import build_datasets
from llib.defaults.main import config as default_config, merge as merge_configs

from llib.data.collective import PartitionSampler
import torch
import smplx
import pickle
import numpy as np
import argparse
import os
from loguru import logger as guru

# set random seeds
import random
SEED = 238492
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg',
        type=str, 
        dest='exp_cfgs', 
        nargs='+', 
        default=None,
        help='The configuration of the experiment'
    )
    parser.add_argument('--exp-opts',
        default=[],
        dest='exp_opts',
        nargs='*',
        help='The configuration of the Detector'
    )
    parser.add_argument('--buddi', 
        type=str, 
        default='demo/diffusion/samples/generate_1000_10_v0/x_starts_smplx.pkl', 
        help='The configuration of the experiment'
    )
    parser.add_argument('--vae',
        type=str,
        default='demo/vae/vae_pred_smpls.pkl',
        help='The configuration of the experiment'
    )
    parser.add_argument('--gt-hi4d',
        type=str,
        default='datasets/processed/Hi4D/processed.pkl',
        help='The configuration of the experiment'
    )
    parser.add_argument(
        '--load-hi4d-data',
        action='store_true',
        help='Load Hi4D data from processed.pkl file'
    )
    parser.add_argument(
        '--load-training-data',
        action='store_true',
        help='Load training data for fid with batch sampler'
    )

    cmd_args = parser.parse_args()

    cfg = merge_configs(cmd_args, default_config)

    return cfg, cmd_args

def fid_featurize(smpls, num_samples=None, device='cuda'):
    
    bs = smpls['global_orient'].shape[0]
    features = {}
    for idx in range(2):
        features.update({
            f'orient_h{idx}': axis_angle_to_rotation6d(smpls['global_orient'][:,idx]),
            f'pose_h{idx}': axis_angle_to_rotation6d(smpls['body_pose'][:,idx].view(bs, -1, 3)).view(bs, -1),
            f'transl_h{idx}': smpls['transl'][:,idx],
            f'shape_h{idx}': torch.cat([smpls['betas'][:,idx], smpls['scale'][:,idx]], dim=-1)
        })

    for k, v in features.items():
        features[k] = v.float().to(device)

    # set transl h0 to yero and h1 relative to h0
    features['transl_h1'] -= features['transl_h0']
    features['transl_h0'] = torch.zeros_like(features['transl_h0'])

    # randomly sample num_samples form each item in features. Make sure index is the same.
    # if num_samples is not None:
    #     sidx = torch.randperm(bs)[:num_samples]
    #     for k, v in features.items():
    #         features[k] = v[sidx]


    return features

def featurized_to_smplx(featurized):
    from llib.utils.threed.conversion import axis_angle_to_rotation6d, rotation6d_to_axis_angle

    bs = featurized['orient_h0'].shape[0]

    humans = []
    for idx in [0,1]:
        humans.append({
            'global_orient': rotation6d_to_axis_angle(featurized[f'orient_h{idx}']),
            'body_pose': rotation6d_to_axis_angle(featurized[f'pose_h{idx}'].view(bs, -1, 6)),
            'transl': featurized[f'transl_h{idx}'],
            'betas': featurized[f'shape_h{idx}'],
        })

    return humans

# load ground-truth data for fid
def load_gt_hi4d_data(
        data_path = 'datasets/processed/Hi4D/processed.pkl',
        split='all',
        train_val_split_path='datasets/processed/Hi4D/train_val_test_split.npz',
):

    assert split in ['train', 'val', 'test', 'all'], \
        'split must be one of [train, val, test, all]'
    
    guru.info(f'Load Hi4D data from {data_path} with split {split}')

    data = pickle.load(open(data_path, 'rb'))
    if split != 'all':
        train_val_split = np.load(train_val_split_path)[split]

    def load_pair_action(data):
        params = {
            'transl': torch.from_numpy(data['smplx_transl_unit']).float(),
            'global_orient': torch.from_numpy(data['smplx_global_orient_unit']).float(),
            'body_pose': torch.from_numpy(data['smplx_body_pose']).float(),
            'betas': torch.from_numpy(data['smplx_betas']).float()[:,:,:10],
            'scale': torch.zeros(data['smplx_betas'].shape[0], 2, 1).float()
        }
        return params
    
    all_params = {
        'global_orient': [],
        'body_pose': [],
        'betas': [],
        'scale': [],
        'transl': []
    }

    for pair_name, pair in data.items():
        if split != 'all':
            if pair_name.replace('pair', '') not in train_val_split:
                continue

        for k, v in pair.items():
            params = load_pair_action(v)
            for k, v in params.items():
                all_params[k].append(v)
    
    # concatenate all items in all_params to tensor
    for k, v in all_params.items():
        all_params[k] = torch.cat(v, dim=0)

    return all_params


# load samples from vae
def load_buddi_generated_samples(path):
    guru.info(f'Load buddi generated samples from {path} ')
    data = pickle.load(open(path, 'rb'))
    return data['final']

# load vae generated samples
def load_vae_generated_samples(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_ae_inout(hi4d_params, num_samples, fid_metric, device='cuda'):

    import smplx 

    model = smplx.create(
        model_path='essentials/body_models',
        model_type='smplx',
        age='kid',
        kid_template_path='essentials/body_models/smil/smplx_kid_template.npy',
        gender='neutral',
        batch_size=num_samples,
    ).to(device)

    def save_meshes(vertices, faces, name, folder='outdebug'):
        import trimesh, os
        os.makedirs(folder, exist_ok=True)
        vertices = vertices.detach().cpu().numpy()
        for idx, vv in enumerate(vertices):
            mesh = trimesh.Trimesh(vv, faces=faces)
            _ = mesh.export(f'{folder}/{idx}_{name}.obj')

    # save meshes of input params
    hi4d_smplx_params = featurized_to_smplx(hi4d_params)
    bodies_h0 = model(**hi4d_smplx_params[0])
    bodies_h1 = model(**hi4d_smplx_params[1])
    save_meshes(bodies_h0.vertices[:10], model.faces, 'h0', folder='outdebug/fid_score_reconstruction_hi4d_orig')
    save_meshes(bodies_h1.vertices[:10], model.faces, 'h1', folder='outdebug/fid_score_reconstruction_hi4d_orig')

    # get reconstruction from autoencoder to check if it's correct
    # this happens inside the autoencoder usually
    if isinstance(hi4d_params, dict):
        x = fid_metric.fid_model.featurizer.embed(hi4d_params)
    x = fid_metric.fid_model.encoder(x)
    x = fid_metric.fid_model.decoder(x)
    hi4d_params_pred = fid_metric.fid_model.featurizer.unembed(x)

    # save meshes of output params
    hi4d_params_pred_smplx = featurized_to_smplx(hi4d_params_pred)
    bodies_h0 = model(**hi4d_params_pred_smplx[0])
    bodies_h1 = model(**hi4d_params_pred_smplx[1])
    save_meshes(bodies_h0.vertices[:10], model.faces, 'h0', folder='outdebug/fid_score_reconstruction_hi4d_pred') 
    save_meshes(bodies_h1.vertices[:10], model.faces, 'h1', folder='outdebug/fid_score_reconstruction_hi4d_pred')

def params_for_fid(params):
    """Takes fid params as input and stacks them together to a single tensor"""
    params_out = None
    num_samples = params['shape_h0'].shape[0]

    params['transl_h1'] -= params['transl_h0']
    params['transl_h0'] = torch.zeros_like(params['transl_h0'])

    for k, v in params.items():
        if params_out is None:
            params_out = v
        # concatenate param_h0 and param_h1 and add to params_out
        params_out = torch.cat([params_out, params[k].reshape(num_samples, -1)], dim=-1)

    return params_out

def fid_on_params(x, y, fid_metric):
    xx = params_for_fid(x).cpu().numpy()
    yy = params_for_fid(y).cpu().numpy()
    m1, s1, m2, s2 = np.mean(xx, axis=0), np.cov(xx, rowvar=False), np.mean(yy, axis=0), np.cov(yy, rowvar=False)
    fid_score_param = fid_metric.calculate_frechet_distance(m1, s1, m2, s2)
    return fid_score_param

def get_fid_and_diversity(
    pred_samples, 
    gt_params, 
    num_samples, 
    fid_metric, 
    diversity_metric, 
    method_name='',
):
    
    ######### fid and diversity for buddi ###########

    pred_params = fid_featurize(pred_samples, num_samples)
    verts = pred_samples.pop('vertices')
    
    if pred_params['pose_h0'].shape[0] > num_samples:
        pred_params = {k: v[:num_samples] for k, v in pred_params.items()}

    #fid_score = fid_metric(pred_params, gt_params)
    fid_score_param = fid_on_params(pred_params, gt_params, fid_metric)

    #verts = pred_samples.pop('vertices')
    verts = verts[:num_samples]
    x, y = torch.split(verts, num_samples // 2, dim=0)
    diversity_score = diversity_metric(x, y)

    #guru.info(f'{method_name} FID score (autoencoder latend): {fid_score}')
    guru.info(f'{method_name} FID score (SMPL params stacked): {fid_score_param}')
    guru.info(f'{method_name} diversity score: {diversity_score}')


def load_training_data(cfg, train_batch_size=512, num_samples=512, device='cuda'):
    train_dataset, _ = build_datasets(
        datasets_cfg=cfg.datasets,
        body_model_type=cfg.body_model.type,  # necessary to load the correct contact maps
    )

    # create / reset sampler
    sampler = PartitionSampler(
        ds_names=train_dataset.dataset_list, 
        ds_lengths=train_dataset.ds_lengths,
        ds_partition=train_dataset.orig_partition,
        shuffle=True,
        batch_size=train_batch_size,
    )

    # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False, # suffling is done in data class
        num_workers=0,
        pin_memory=False,
        drop_last=False, # was false for bs 64
        sampler=sampler
    )

    train_samples = {
        'global_orient': [],
        'body_pose': [],
        'betas': [],
        'scale': [],
        'transl': [],
    }

    counter = 0
    for batch_idx, batch in enumerate(train_loader):
        if counter > num_samples:
            break
        for k, v in train_samples.items():
            train_samples[k].append(batch[f'pgt_{k}'])
        counter += train_batch_size

    # cat train samples
    for k, v in train_samples.items():
        train_samples[k] = torch.cat(v, dim=0).to(device)

    train_samples['transl'][:,1,:] -= train_samples['transl'][:,0,:]
    train_samples['transl'][:,0,:] = 0
    train_samples['betas'] = torch.cat(
        [train_samples.pop('betas'), train_samples.pop('scale')], dim=-1
    )
    
    # get smplx model with batch size like train samples
    ccc = 10
    train_body_model = smplx.create(
        model_path = 'essentials/body_models',
        model_type='smplx',
        age='kid',
        kid_template_path='essentials/body_models/smil/smplx_kid_template.npy',
        gender='neutral',
        batch_size=ccc,
    ).to(device)

    # get vertices for train samples in batches of size 1000
    vertices = []
    for idx in range(0, len(train_samples['global_orient']), ccc):
        if idx + ccc > len(train_samples['global_orient']):
            continue

        with torch.no_grad():
            v0 = train_body_model(**{k: v[idx:idx+ccc,0,:] for k, v in train_samples.items()}).vertices
            v1 = train_body_model(**{k: v[idx:idx+ccc,1,:] for k, v in train_samples.items()}).vertices
            vv = torch.cat([v0.unsqueeze(1), v1.unsqueeze(1)], dim=1).detach().cpu()
            vertices.append(vv)
            
    train_samples['vertices'] = torch.cat(vertices, dim=0)
    betas = train_samples['betas'].clone()
    train_samples['scale'] = betas[:,:,10:]
    train_samples['betas'] = betas[:,:,:10]

    train_samples = {k: v[:num_samples] for k, v in train_samples.items()}

    return train_samples

# build fid metrics
def main(
        cfg, # eval config
        args, # cmd_args
        num_samples='max',
        device='cuda',
        debug=False,
        use_shape_params=False
):
    
    num_samples = 2 * num_samples
    hns = num_samples // 2

    # create metrics
    fid_metric = GenFID()
    diversity_metric = GenDiversity()

   
    if args.load_training_data:
        train_data = load_training_data(
            cfg, train_batch_size=512, num_samples=num_samples, device=device
        )
        train_num_samples = len(train_data['vertices'])
        guru.info(f'Loaded {train_num_samples} Training samples')

        train_data_01 = {k: v[:hns] for k, v in train_data.items()}
        train_data_02 = {k: v[hns:] for k, v in train_data.items()}


    buddi_samples = load_buddi_generated_samples(args.buddi)
    buddi_num_samples = len(buddi_samples['global_orient'])
    guru.info(f'Loaded {buddi_num_samples} buddi samples')

    if os.path.exists(args.vae):
        vae_samples = load_vae_generated_samples(args.vae)
        vae_num_samples = len(vae_samples['global_orient'])
        guru.info(f'Loaded {vae_num_samples} VAE samples')
    else:
        # set to buddi count, but ignore vae later
        vae_num_samples = buddi_num_samples

    ###################### COMPUTE METRICS ##################################
    # if args.load_hi4d_data:
    #     # train_data = hi4d_data

    if args.load_training_data:
        # pick number of samples to max
        if num_samples == 'max':
            num_samples = min(buddi_num_samples, train_num_samples, vae_num_samples)
            guru.info(f'Picked {num_samples} samples')

        # load ground-truth data for fid
        train_params_01 = fid_featurize(train_data_01, hns)
        train_params_02 = fid_featurize(train_data_02, hns)

        if debug:
            save_ae_inout(train_params_01, num_samples, fid_metric, device)
        
        ######### FID of Training data against itself via two random samples #########        
        get_fid_and_diversity(train_data_01, train_params_02, hns, fid_metric, diversity_metric, method_name='train')
        get_fid_and_diversity(buddi_samples, train_params_02, hns, fid_metric, diversity_metric, method_name='buddi')
        
        if os.path.exists(args.vae):
            get_fid_and_diversity(vae_samples, train_params_02, hns, fid_metric, diversity_metric, method_name='vae')


if __name__ == "__main__":
    cfg, cmd_args = parse_args()
    main(cfg, cmd_args, num_samples=8000) #2048)