import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from .renderer import Renderer
from .plotter import Plotter

@dataclass 
class Visualization:
    renderer: Renderer = Renderer()
    plotter: Plotter = Plotter()

conf = OmegaConf.structured(Visualization)