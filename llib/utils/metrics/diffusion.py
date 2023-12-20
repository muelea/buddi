
from typing import Iterator
import torch.nn as nn
import torch
import numpy as np
import scipy
import os
from torch.nn.modules.module import Module
from llib.utils.metrics.points import PointError
from llib.utils.threed.distance import pcl_pcl_pairwise_distance
from llib.utils.threed.intersection import winding_numbers
#from tsnecuda import TSNE
from omegaconf import OmegaConf
from loguru import logger as guru
from llib.defaults.main import (
    config as default_config,
    merge as merge_configs
)
# import build_model function
from llib.models.build import build_model

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

class tSNE(nn.Module):
    def __init__(
        self,
        name='v2v',
    ):
        super(tSNE, self).__init__()
        """
        vertices: Numpy array of size (B, 2, N, 3) containing the ground truth data
        """


    def forward(self, x):
        """
        Fast tSNE implementation using tsnecuda
        x: Numpy array of size (B, F) containing the vertices
        """

        X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x)

        return X_embedded

class GenContactIsect(nn.Module):
    def __init__(
            self,
            name='GenContactIsect',
            contact_thres=0.01,
            ww_thres=0.99,
    ):
        super(GenContactIsect, self).__init__()
        """
            Class to get statistics of contact and intersection between two sets of vertices

            contact_thres: threshold for contact distance. If distance between two vertices 
            is < contact_thres, then they are considered in contact.
            ww_thres: threshold for winding number. If winding number of a vertex is > ww_thres
            then it is considered inside the mesh.
        """

        self.ww_thres = ww_thres
        self.contact_thres = contact_thres


    def close_mouth(self, v):
            mv = torch.mean(v[:,self.vert_ids_wt,:], 1, keepdim=True)
            v = torch.cat((v, mv), 1)
            return v

    def to_lowres(self, v, n=100):
        lrm = self.low_res_mesh[n]
        v = self.close_mouth(v)
        v = v[:,lrm['smplx_vid'],:]
        t = v[:,lrm['faces'].astype(np.int32),:]
        return v, t

    def get_intersecting_verts(self, x):
        """
            Computes the inside mask for the given vertices
            x: Numpy array of size (B, N, 3) containing the vertices
        """
        _, t = self.to_lowres(x)
        interior = winding_numbers(x, t).ge(self.ww_thres)
        return interior

    def forward(self, x, y):
        """
            x: Numpy array of size (B, N, 3) containing the first set of vertices
            y: Numpy array of size (B, N, 3) containing the second set of vertices
        """

        interior_v1 = self.get_intersecting_verts(x)
        interior_v2 = self.get_intersecting_verts(y)

        # compute distance between v1 and v2
        v1v2 = pcl_pcl_pairwise_distance(
            x, y, squared=False)

        # minimum distance between v1 and v2
        min_dist = sum(v1v2 < self.contact_thres).item()

        # intersection distances from v1 vertices to v2
        max_val, mean_val, median_val = 0.0, 0.0, 0.0
        if interior_v1.any():
            v1_to_v2 = v1v2[:,interior_v1,:].min(2)[0]
            max_val = v1_to_v2.max().item()
            mean_val = v1_to_v2.mean().item()
            median_val = v1_to_v2.median().item()

        # intersection distances from v2 vertices to v1
        max_val2, mean_val2, median_val2 = 0.0, 0.0, 0.0
        if interior_v2.any():
            v2_to_v1 = v1v2[interior_v2,:,:].min(1)[0]
            max_val2 = v2_to_v1.max().item()
            mean_val2 = v2_to_v1.mean().item()
            median_val2 = v2_to_v1.median().item()

        stats = {
            'min_dist': min_dist,
            'max_v1_in_v2': max_val,
            'mean_v1_in_v2': mean_val,
            'median_v1_in_v2': median_val,
            'max_v2_in_v1': max_val2,
            'mean_v2_in_v1': mean_val2,
            'median_v2_in_v1': median_val2,
        }

        return stats
        
class GenDiversity(nn.Module):
    def __init__(
            self,
            name='GenDiversity',
    ):
        super(GenDiversity, self).__init__()

    def forward(self, x, y):
        """
            Computes the diversity between the two sets of vertices
            x: Numpy array of size (B, 2, N, 3) containing the first set of vertices
            y: Numpy array of size (B, 2, N, 3) containing the second set of vertices
        """

        diversity = torch.sqrt(torch.pow(x - y, 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1)
        diversity = diversity.mean() # average over batch
        return diversity

class GenFID(nn.Module):
    def __init__(
            self,
            name='GenFID',
            fid_model_path='essentials/buddi/fid_model.pt',
            fid_model_cfg='essentials/buddi/fid_model_config.yaml',
    ):
        super(GenFID, self).__init__()

        self.fid_model_type = None

        # load the fid model with pytorch
        if os.path.exists(fid_model_path) and os.path.exists(fid_model_cfg):
            fid_model_cfg = OmegaConf.load(fid_model_cfg)
            self.fid_model_type = fid_model_cfg.model.regressor.type
            self.fid_model = build_model(fid_model_cfg.model.regressor).to(fid_model_cfg.device)
            checkpoint = torch.load(fid_model_path)
            self.fid_model.load_state_dict(checkpoint['model'], strict=False)
            self.fid_model.eval()
            guru.info('Loaded FID model from {}'.format(fid_model_path))


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.

        Taken from here: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def compute_statistics(self, x):
        """
            Computes the mean and covariance of the input features
            x: Numpy array of size (B, F) containing the features
        """

        # check if x is numpy array
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # compute mean and covariance
        mu = np.mean(x, axis=0)
        sigma = np.cov(x, rowvar=False)

        return mu, sigma

    def forward(self, x, y):
        """
            Computes the FID between the two sets of vertices
            x: Numpy array of size (B, F) containing the first params or a dict with params as keys (orient, pose, shape, transl)
            y: Numpy array of size (B, F) containing the second params or a dict with params as keys (orient, pose, shape, transl)
        """
       
        # check if x / y is a fict and if so get the params
        if isinstance(x, dict):
            x = self.fid_model.featurizer.embed(x)
        if isinstance(y, dict):
            y = self.fid_model.featurizer.embed(y)

        if self.fid_model_type == 'autoencoder_mlp':
            x = self.fid_model.encoder(x).detach().cpu().numpy() 
            y = self.fid_model.encoder(y).detach().cpu().numpy()


        m1, s1 = self.compute_statistics(x)
        m2, s2 = self.compute_statistics(y)
        fid = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid
