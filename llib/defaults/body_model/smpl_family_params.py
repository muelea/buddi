import os
import torch
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
import torch 

@dataclass
class BodyPose:
    create: bool = True 
    value = None

@dataclass
class Shape:
    dim: int = 10
    create: bool = True 
    value = None

@dataclass 
class GlobalOrient:
    create: bool = True 
    value = None

@dataclass
class Translation:
    create: bool = True 
    value = None

@dataclass 
class HandPose:
    use_pca: bool = True
    num_pca_comps: int = 6
    flat_hand_mean: bool = False
    create: bool = True 
    value = None

@dataclass
class Expression:
    dim: int = 10
    create: bool = True 
    value = None

@dataclass
class Jaw:
    create: bool = True 
    value = None

@dataclass
class Eye:
    create: bool = True 
    value = None