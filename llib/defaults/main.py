import os
from omegaconf import OmegaConf 
from dataclasses import dataclass

from .datasets.main import conf as datasets_conf, Datasets
from .body_model.main import conf as bodymodel_conf, BodyModel
from .model.main import conf as model_conf, Model
from .logging.main import conf as logging_conf, Logging
from .camera.main import conf as camera_conf, Camera
from .visualization.main import conf as visu_conf, Visualization
from .training.train import conf as training_conf, Training
from .training.eval import conf as evaluation_conf, Evaluation


def merge(cmd_args, default_config):
    """
    Merge omegaconf file with command line arguments
    config files and command line arguments
    """

    cfg = default_config.copy()
    
    if cmd_args.exp_cfgs:
        for exp_cfg in cmd_args.exp_cfgs:
            if exp_cfg:
                cfg.merge_with(OmegaConf.load(exp_cfg))

    if cmd_args.exp_opts:
        cfg.merge_with(OmegaConf.from_cli(cmd_args.exp_opts))
    
    return cfg

@dataclass
class Config:

    batch_size: int = 1
    device: str = 'cuda'

    # body model
    body_model: BodyModel = bodymodel_conf

    # datasets
    datasets: Datasets = datasets_conf

    # camera
    camera: Camera = camera_conf 

    # model
    model: Model = model_conf

    # training
    training: Training = training_conf
    evaluation: Evaluation = evaluation_conf

    # renderer
    visualization: Visualization = visu_conf            

    # output
    logging: Logging = logging_conf

config = OmegaConf.structured(Config)
