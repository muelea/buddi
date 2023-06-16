import torch.nn as nn
import numpy as np
import torch
from llib.losses.build import build_loss
from llib.utils.threed.conversion import batch_rodrigues
from llib.utils.metrics.build import build_metric

class EvalModule(nn.Module):
    def __init__(
        self,
        eval_cfgs,
        body_model_type='smplx'
    ):
        super(EvalModule, self).__init__()

        self.cfg = eval_cfgs

        self.per_person_metrics = ['v2v', 'mpjpe', 'pa_mpjpe']
        self.num_humans = 2

        self.accumulator = {} # dict to store all the metrics
        self._init_metrics()     

        self.tb_output = None
    
    def reset(self):
        self.accumulator = {}
        self._init_metrics()

    def _init_metrics(self):
        # add member variables for each metric
        for name in self.cfg.metrics:
            metric = build_metric(self.cfg[name])
            setattr(self, f'{name}_func', metric)   

        # fill accumulator dict
        for name in self.cfg.metrics:
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

    def accumulate(self, metric_name, array):
        curr_array = self.accumulator[metric_name]
        self.accumulator[metric_name] = np.concatenate((curr_array, array), axis=0) \
            if curr_array.size else array

    def forward(
        self, 
        est_smpl=None,
        tar_smpl=None,
        est_contact_map_heat=None,
        tar_contact_map_heat=None,
        est_contact_map_binary_smpl=None,
        est_contact_map_binary_token=None,
        tar_contact_map_binary=None,
    ):  

        for name in self.cfg.metrics:

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
        self.ckpt_metric_value = torch.tensor(self.accumulator[self.cfg.checkpoint_metric]).mean()

        # add metric to tensorboard
        accumulator_keys = list(self.accumulator.keys())
        for key in accumulator_keys:
            self.accumulator[key] = torch.tensor(self.accumulator[key]).mean()