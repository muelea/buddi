
import torch.nn as nn
import torch
import numpy as np
from llib.utils.metrics.points import PointError

class GenerationGraph(nn.Module):
    def __init__(
        self,
        target, 
        metric_name='v2v',
        alignment='procrustes',
    ):
        super(GenerationGraph, self).__init__()
        """
        vertices: Numpy array of size (B, 2, N, 3) containing the ground truth data
        """
        self.target = target
        self.metric = PointError(metric_name, alignment)
        
    def forward(self, pred_vertices):
        """
            Selects the best match for each predicted vertex from gt dataset
            and computes the returns the minimum v2v error between the two.
        """

        output = {
            'mapping': [],
            'min_error': []
        }

        # for each item in predicted vertices find the gt verts with smalles v2v error
        for pv in pred_vertices:
            errors = []
            for gt in self.target:
                errors.append(self.metric(pv[None], gt[None]).mean())
            output['mapping'].append(np.argmin(np.array(errors)))
            output['min_error'].append(min(errors))

        return output
        

        