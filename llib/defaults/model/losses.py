import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class Loss:
    type: str = ""
    weight: List[float] = field(default_factory=lambda: [0.0])


@dataclass
class L2Loss(Loss):
    squared: bool = False
    translated: bool = False
    weighted: bool = False
    d1_aggregation: str = "sum"


@dataclass
class ContactLoss(Loss):
    region_aggregation_type: str = "sum"
    r2r_dist_type: str = "min"
    squared_dist: bool = True
    region_to_vertex: str = "essentials/contact/flickrci3ds_r75_rid_to_smplx_vid.pkl"


@dataclass
class GeneralContactLoss(Loss):
    region_aggregation_type: str = "sum"
    r2r_dist_type: str = "min"
    squared_dist: bool = True
    body_model_utils_folder: str = "essentials/body_model_utils"
    model_type: str = "smplx"


@dataclass
class MaxMixturePrior(Loss):
    prior_folder: str = "essentials/priors"
    num_gaussians: int = 8
    epsilon: float = 1e-16
    use_merged: bool = True


@dataclass
class CMap(Loss):
    r2r_dist_type: str = "test"


@dataclass
class AnnealLoss(Loss):
    anneal_start: int = 0
    anneal_end: int = -1


@dataclass
class Losses:
    debug: bool = False
    keypoint2d: L2Loss = L2Loss(type="l2", weighted=True, squared=True)
    init_pose: L2Loss = L2Loss(type="l2", squared=True)
    init_shape: L2Loss = L2Loss(type="l2", squared=True)
    init_transl: L2Loss = L2Loss(type="l2", squared=True)
    kl: AnnealLoss = AnnealLoss(type="")
    pseudogt_pose: L2Loss = L2Loss(type="l2", squared=True)
    pseudogt_shape: L2Loss = L2Loss(type="l2", squared=True)
    pseudogt_transl: L2Loss = L2Loss(type="l2", squared=True)
    pseudogt_v2v: L2Loss = L2Loss(type="l2", squared=True)
    pseudogt_j2j: L2Loss = L2Loss(type="l2", squared=True)
    hhc_contact: ContactLoss = ContactLoss(type="hhcmap")
    hhc_contact_general: GeneralContactLoss = GeneralContactLoss(type="hhcgen")
    pose_prior: MaxMixturePrior = MaxMixturePrior(type="gmm")
    shape_prior: L2Loss = L2Loss(type="l2", squared=True)
    ground_plane: L2Loss = L2Loss(type="l2", squared=True)
    cmap: CMap = CMap(type="cmap")
    cmap_heat_smpl: Loss = Loss(type="")
    cmap_heat_token: Loss = Loss(type="")
    cmap_binary_smpl: Loss = Loss(type="")
    cmap_binary_token: Loss = Loss(type="")
    diffusion_prior_orient: L2Loss = L2Loss(type="l2", squared=True)
    diffusion_prior_pose: L2Loss = L2Loss(type="l2", squared=True)
    diffusion_prior_shape: L2Loss = L2Loss(type="l2", squared=True)
    diffusion_prior_scale: L2Loss = L2Loss(type="l2", squared=True)
    diffusion_prior_transl: L2Loss = L2Loss(type="l2", squared=True)
    diffusion_prior_v2v: L2Loss = L2Loss(type="l2", squared=True)
    vae_prior_latent: L2Loss = L2Loss(type="l2", squared=True)
