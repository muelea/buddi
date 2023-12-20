import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import List
from dataclasses import field
from .datasets import *
from .utils import Augmentation, Processing

# Steps to add a new dataset:
# 1) add name and composition to train_names/train_composition/val_names
# 2) in list each dataset, add name: Name = Name()
# 3) add dataset in datasets.py

@dataclass 
class Datasets:

    # image processing
    processing: Processing = Processing()

    # training data
    train_names: List[str] = field(default_factory=lambda: ['flickrci3ds', 'flickrci3dc', 'chi3d'])
    train_composition: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    augmentation: Augmentation = Augmentation(
        use=True, mirror=0.5, noise=0.4, rotation=30, scale=0.25
    )

    # validation data
    val_names: List[str] = field(default_factory=lambda: ['flickrci3ds', 'flickrci3dc', 'chi3d'])
    
    # test data 
    test_names: List[str] = field(default_factory=lambda: [])
    
    # list all datasets here
    flickrci3dsd: FlickrCI3D_SignaturesDownstream = FlickrCI3D_SignaturesDownstream()
    flickrci3ds: FlickrCI3D_Signatures = FlickrCI3D_Signatures()
    flickrci3ds_adult: FlickrCI3D_Signatures = FlickrCI3D_Signatures(adult_only=True)
    flickrci3ds_child: FlickrCI3D_Signatures = FlickrCI3D_Signatures(child_only=True)
    flickrci3dc: FlickrCI3D_Classification = FlickrCI3D_Classification()
    chi3d: CHI3D = CHI3D()
    hi4d: HI4D = HI4D()
    demo: Demo = Demo()


conf = OmegaConf.structured(Datasets)