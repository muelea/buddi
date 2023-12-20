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
class GenDiversity:
    name: str = 'GenDiversity'

@dataclass
class GenFID:
    fid_model_path: str = ''
    fid_model_cfg: str = ''
    name: str = 'GenFID'

@dataclass
class GenContactIsect:
    name: str = 'GenContactIsect'

@dataclass
class GentSNE:
    name: str = 'GentSNE'

@dataclass 
class Evaluation:
    checkpoint_metric: str = 'total_loss' # this is the value which will be added to the checkpoint filename

    # the metrics computed during validation and evaluation
    metrics: List[str] = field(default_factory=lambda: []) #['mpjpe', 'pa_mpjpe', 'pairwise_pa_mpjpe', 'cmap_dist'])
    per_person_metrics: List[str] = field(default_factory=lambda: ['v2v','mpjpe', 'scale_mpjpe', 'pa_mpjpe'])
    generative_metrics: List[str] = field(default_factory=lambda: []) #['gen_diversity', 'gen_fid', 'gen_contact_and_isect', 'gen_tsne'])

    # the number of samples to generate for the generative metrics 
    # for FID and diversity this number will be split between two halves
    num_samples: int = 200

    # metrics
    v2v: PointError = PointError(alignment='root')
    mpjpe: PointError = PointError(alignment='root')
    scale_mpjpe: PointError = PointError(alignment='scale')
    pa_mpjpe: PointError = PointError(alignment='procrustes')
    pairwise_pa_mpjpe: PointError = PointError(alignment='procrustes')
    cmap_dist: ContactMapDistError = ContactMapDistError()
    cmap_iou: ContactIOU = ContactIOU()
    gen_diversity: GenDiversity = GenDiversity()
    gen_fid: GenFID = GenFID()
    gen_contact_and_isect: GenContactIsect = GenContactIsect()
    gen_tsne: GentSNE = GentSNE()

conf = OmegaConf.structured(Evaluation)