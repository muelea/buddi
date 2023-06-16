import torch.nn as nn
from .l2 import L2Loss
from .contact import MinDistLoss, ContactMapLoss, GeneralContactLoss
from .contactmap import ContactMapEstimationLoss
from .gmm import MaxMixturePrior

class Placeholder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return None


def build_loss(loss_cfg, body_model_type='smplx'):
    loss_type = loss_cfg.type
    if loss_type == 'l2':
        loss = L2Loss(**loss_cfg)
    elif loss_type == 'hhcmap':
        loss = ContactMapLoss(**loss_cfg)
    elif loss_type == 'hhcdistmin':
        loss = MinDistLoss(**loss_cfg)
    elif loss_type == 'hhcgen':
        loss = GeneralContactLoss(**loss_cfg)
    elif loss_type == 'gmm':
        loss = MaxMixturePrior(model_type=body_model_type, **loss_cfg)
    elif loss_type == 'cmap':
        loss = ContactMapEstimationLoss(**loss_cfg)
    elif loss_type == '':
        loss = Placeholder()
    else:
        raise ValueError(f'Loss {loss_type} not implemented')
    return loss
