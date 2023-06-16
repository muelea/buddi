import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass
class PointError:
    name: str = 'PointError'
    alignment: str = 'root'

@dataclass
class ContactMapDistError:
    name: str = 'ContactMapDistError'
    region_to_vertex: str = 'essentials/contact/flickrci3ds_r75_rid_to_smplx_vid.pkl'

@dataclass
class ContactIOU:
    name: str = 'ContactIOU'

@dataclass 
class Evaluation:
    checkpoint_metric: str = 'total_loss' # this is the value which will be added to the checkpoint filename

    # the metrics computed during validation and evaluation
    metrics: List[str] = field(default_factory=lambda: ['mpjpe', 'pa_mpjpe', 'pairwise_pa_mpjpe', 'cmap_dist'])

    # metrics
    v2v: PointError = PointError(alignment='root')
    mpjpe: PointError = PointError(alignment='root')
    scale_mpjpe: PointError = PointError(alignment='scale')
    pa_mpjpe: PointError = PointError(alignment='procrustes')
    pairwise_pa_mpjpe: PointError = PointError(alignment='procrustes')
    cmap_dist: ContactMapDistError = ContactMapDistError()
    cmap_iou: ContactIOU = ContactIOU()

conf = OmegaConf.structured(Evaluation)