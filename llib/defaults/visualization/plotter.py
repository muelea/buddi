import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass

@dataclass
class PlotStyle:
    type: str = 'latex'

@dataclass
class Plotter:
    iw: int = 224
    ih: int = 224
    style: PlotStyle = PlotStyle()
