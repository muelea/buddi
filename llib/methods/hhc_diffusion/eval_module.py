import torch.nn as nn
import numpy as np
import torch
from llib.losses.build import build_loss
from llib.utils.threed.conversion import batch_rodrigues
from llib.utils.metrics.build import build_metric
from llib.utils.threed.conversion import (
    axis_angle_to_rotation6d,
)

class EvalModule(nn.Module):
    def __init__(
        self,
        eval_cfgs,
        body_model_type='smplx'
    ):
        super(EvalModule, self).__init__()

        self.cfg = eval_cfgs

        self.per_person_metrics = self.cfg.per_person_metrics #['v2v', 'scale_mpjpe','mpjpe', 'pa_mpjpe']
        self.num_humans = 2

        # metrics for reconstruction        
        self.metrics = self.cfg.metrics

        # metrics for generation only
        self.generative_metrics = self.cfg.generative_metrics # ['fid', 'diversity', 'isect', 'contacts']
        self.num_samples = self.cfg.num_samples
        assert self.num_samples % 2 == 0, 'max_samples must be even'

        self.all_metrics = self.metrics + self.generative_metrics

        self.accumulator = {} # dict to store all the metrics
        self._init_metrics()     

        self.tb_output = None
    
    def reset(self):
        self.accumulator = {}
        self._init_metrics()

    def _init_metrics(self):
        # add member variables for each metric
        for name in self.all_metrics:
            metric = build_metric(self.cfg[name])
            setattr(self, f'{name}_func', metric)   

        # fill accumulator dict
        for name in self.all_metrics:
            if name in self.per_person_metrics:
                for i in range(self.num_humans): 
                    self.accumulator[f'{name}_{i}'] = np.array([])
            else:  
                if name == 'cmap_iou':
                    self.accumulator['cmap_iou_token'] = np.array([])
                    self.accumulator['fscore_token'] = np.array([])
                    self.accumulator['precision_token'] = np.array([])
                    self.accumulator['recall_token'] = np.array([]) 
                    self.accumulator['cmap_iou_smpl'] = np.array([])
                    self.accumulator['fscore_smpl'] = np.array([])
                    self.accumulator['precision_smpl'] = np.array([])
                    self.accumulator['recall_smpl'] = np.array([]) 
                else:
                    self.accumulator[name] = np.array([])       

        # add total_loss to accumulator
        self.accumulator['total_loss'] = np.array([])       

    def accumulate(self, metric_name, array):
        curr_array = self.accumulator[metric_name]
        self.accumulator[metric_name] = np.concatenate((curr_array, array), axis=0) \
            if curr_array.size else array

    def forward_generative_metrics(self, gen_smpls, real_smpls=None, shuffle=True):
        """
            gen_smpls: tensor of generated smpls of size (num_samples, num_humans)
        """

        genh0, genh1 = gen_smpls 
        bs = genh0.global_orient.size()[0]
        gen_verts = torch.cat([genh0.vertices.unsqueeze(1), genh1.vertices.unsqueeze(1)], dim=1)

        if real_smpls is not None:
            real_verts = torch.cat([real_smpls[0].vertices.unsqueeze(1), real_smpls[1].vertices.unsqueeze(1)], dim=1)

        def fid_features(smpls):
            features = {}
            for idx in range(2):
                features.update({
                    f'orient_h{idx}': axis_angle_to_rotation6d(smpls[idx].global_orient),
                    f'pose_h{idx}': axis_angle_to_rotation6d(smpls[idx].body_pose.view(bs, -1, 3)).view(bs, -1),
                    f'transl_h{idx}': smpls[idx].transl,
                    f'shape_h{idx}': torch.cat([smpls[idx].betas, smpls[idx].scale], dim=-1)
                })
            return features

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
        #assert gen_smpls.size()[0] == self.num_samples, \
        #    'num_samples must be equal to gen_smpls.size()[0]'
        
        # randomly suffle the generated smpls along num_samples dimension
        #if shuffle:
        #    gen_smpls = gen_smpls[torch.randperm(gen_smpls.size()[0])]

        for name in self.generative_metrics:
            metric = getattr(self, f'{name}_func')
            if name == 'gen_diversity':
                # split gen_samples along num_samples dimension in two halves
                # and compute the metric between the two halves
                def prep_verts(verts):
                    x = verts[:self.num_samples // 2]
                    y = verts[self.num_samples // 2:]
                    x = x.reshape(x.size()[0], -1, x.size()[-1])
                    y = y.reshape(y.size()[0], -1, y.size()[-1])
                    return x, y
                x, y = prep_verts(gen_verts)
                error = metric(x, y)
                # real vertices error: 0.7801 for CHI3D val set
                self.accumulate(name, error.cpu().numpy()[None])   
            elif name == 'gen_fid':
                gen_features = fid_features(gen_smpls)
                gen_features = params_for_fid(gen_features)
                real_features = fid_features(real_smpls)
                real_features = params_for_fid(real_features)
                error = metric(gen_features, real_features)
                self.accumulate(name, np.array([error]))         
            elif name == 'gen_contact_and_isect': # for contact and isect
                pass
                #errors = metric(gen_verts[:,0], gen_verts[:,1])
                #self.accumulate(name, errors.cpu().numpy())


    def forward(
        self, 
        est_smpl=None,
        tar_smpl=None,
        est_contact_map_heat=None,
        tar_contact_map_heat=None,
        est_contact_map_binary_smpl=None,
        est_contact_map_binary_token=None,
        tar_contact_map_binary=None,
        t_type=''
    ):  

        for name in self.metrics:

            metric = getattr(self, f'{name}_func') # metric class / function
            if name in self.per_person_metrics:
                for i in range(self.num_humans):
                    if name == 'v2v':
                        in_points, tar_points = est_smpl[i].vertices, tar_smpl[i].vertices
                    elif name in ['mpjpe', 'pa_mpjpe']:
                        in_points, tar_points = est_smpl[i].joints, tar_smpl[i].joints                   
                    errors = metric(in_points.cpu().numpy(), tar_points.cpu().numpy())
                    self.accumulate(f'{name}_{i}', errors)
            else:
                if name == 'cmap_dist':
                    errors = metric(est_smpl[0].vertices, est_smpl[1].vertices, tar_contact_map_binary)
                    self.accumulate(name, errors.cpu().numpy())
                elif name == 'pairwise_pa_mpjpe':
                    in_points = torch.cat([est_smpl[i].joints for i in range(self.num_humans)], dim=1)
                    tar_points = torch.cat([tar_smpl[i].joints for i in range(self.num_humans)], dim=1)
                    errors = metric(in_points.cpu().numpy(), tar_points.cpu().numpy())
                    self.accumulate(name, errors)
                elif name == 'cmap_iou':
                    contact_from = ['smpl', 'token'] if est_contact_map_binary_token is not None else ['smpl']
                    for nn in contact_from:
                        est_contact_map = est_contact_map_binary_smpl if nn == 'smpl' else est_contact_map_binary_token
                        errors, precision, recall, fsore = metric(est_contact_map, tar_contact_map_binary)
                        self.accumulate(f'cmap_iou_{nn}', errors.cpu().numpy())
                        self.accumulate(f'fscore_{nn}', fsore.cpu().numpy())
                        self.accumulate(f'precision_{nn}', precision.cpu().numpy())
                        self.accumulate(f'recall_{nn}', recall.cpu().numpy()) 

    def final_accumulate_step(self):
        if self.cfg.checkpoint_metric in self.accumulator.keys():
            self.ckpt_metric_value = torch.tensor(self.accumulator[self.cfg.checkpoint_metric]).mean()
        else:
            self.ckpt_metric_value = torch.tensor(0.0)

        # add metric to tensorboard
        accumulator_keys = list(self.accumulator.keys())
        for key in accumulator_keys:
            self.accumulator[key] = torch.tensor(self.accumulator[key]).mean()