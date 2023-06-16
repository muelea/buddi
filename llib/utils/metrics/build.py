from llib.utils.metrics.alignment import * 
from llib.utils.metrics.points import PointError
from llib.utils.metrics.contact import ContactMapDistError, ContactIOU

def build_metric(cfg):
    if cfg.name == 'PointError':
        return PointError(**cfg)
    elif cfg.name == 'ContactMapDistError':
        return ContactMapDistError(**cfg)
    elif cfg.name == 'ContactIOU':
        return ContactIOU(**cfg)
    else:
        raise ValueError(f'Unknown metric type: {cfg.name}')