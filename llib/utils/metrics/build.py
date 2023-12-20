from llib.utils.metrics.alignment import * 
from llib.utils.metrics.points import PointError
from llib.utils.metrics.contact import ContactMapDistError, ContactIOU
from llib.utils.metrics.diffusion import GenDiversity, GenFID, GenContactIsect, tSNE

def build_metric(cfg):
    if cfg.name == 'PointError':
        return PointError(**cfg)
    elif cfg.name == 'ContactMapDistError':
        return ContactMapDistError(**cfg)
    elif cfg.name == 'ContactIOU':
        return ContactIOU(**cfg)
    elif cfg.name == 'GenDiversity':
        return GenDiversity(**cfg)
    elif cfg.name == 'GenFID':
        return GenFID(**cfg)
    elif cfg.name == 'GenContactIsect':
        return GenContactIsect(**cfg)
    elif cfg.name == 'GentSNE':
        return tSNE(**cfg)
    else:
        raise ValueError(f'Unknown metric type: {cfg.name}')