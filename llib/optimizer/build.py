import torch

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
    else:
        raise NotImplementedError
    
    return optimizer