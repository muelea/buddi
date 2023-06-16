import pickle
import torch 
import numpy as np
import torch.nn as nn
import os.path as osp
from llib.utils.threed.distance import pcl_pcl_pairwise_distance
from llib.utils.threed.intersection import winding_numbers


class ContactMapEstimationLoss(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        """
        Aggregated distance between multiple point clouds.
        """

    def forward(self, x, y):
        loss = ((x - y)**2).sum()
        return loss
