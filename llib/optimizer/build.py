import torch
from loguru import logger as guru

def build_optimizer(cfg, optimizer_type, params):
    """
    Build optimizer from config.
    Parameters
    ----------
    cfg: cfg
        The configuration of the optimizer.
    optimizer_type: str
        The type of the optimizer to build.
    params: list
        The parameters to optimize.
    """
    
    # setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfg.adam.lr, 
            weight_decay=cfg.adam.weight_decay
        )
    elif optimizer_type.lower() == 'lbfgs':
        optimizer = torch.optim.LBFGS(
            params=params,
            lr=cfg.lbfgs.lr
        )
    else:
        raise NotImplementedError
    
    return optimizer