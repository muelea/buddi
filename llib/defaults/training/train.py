import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass

@dataclass 
class Training:
    max_epochs: int = 100000000
    max_duration: float = float('inf')
    num_workers: int = 8
    pin_memory: bool = False # pin memory for dataloader
    shuffle_train: bool = True # shuffle training data
    pretrained: str = '' # load weights of a pretrained model
    clip_grad_norm: float = 0.0 # clip gradient norm
    train: bool = True 
    eval_val: bool = False # evaluate on validation set
    eval_test: bool = False # evaluate on test set

conf = OmegaConf.structured(Training)