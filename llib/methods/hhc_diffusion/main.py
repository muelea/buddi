# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

#import os
#import json
#from utils.parser_util import train_args
import argparse

from llib.models.build import build_model
from llib.optimizer.build import build_optimizer
from llib.data.build import build_datasets
from llib.logging.logger import Logger
from llib.training.diffusion_trainer import Trainer
from llib.bodymodels.build import build_bodymodel 
from llib.cameras.build import build_camera
from llib.visualization.renderer import Pytorch3dRenderer
from loss_module import LossModule

from llib.models.diffusion.build import build_diffusion

from llib.defaults.main import (
    config as default_config,
    merge as merge_configs
)

from train_module import TrainModule
from eval_module import EvalModule

import torch
torch.autograd.set_detect_anomaly(True)

import numpy as np
import random
SEED = 238492
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg', 
        type=str, dest='exp_cfgs', nargs='+', default=None, 
        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
        nargs='*', help='The configuration of the Detector') 

    cmd_args = parser.parse_args()

    cfg = merge_configs(cmd_args, default_config)

    return cfg


def train(cfg):
    # create logger and write vconfig file
    logger = Logger(cfg)
    
    # create datasets
    train_dataset, val_dataset = build_datasets(
        datasets_cfg=cfg.datasets,
        body_model_type=cfg.body_model.type # necessary to load the correct contact maps
    )
    
    # build regressor used to predict diffusion params
    regressor = build_model(cfg.model.regressor).to(cfg.device)

    # build diffusion process
    diffusion = build_diffusion(**cfg.model.diffusion)

    # load body models for human1 and human2
    body_model = build_bodymodel(
        cfg=cfg.body_model, 
        batch_size=cfg.batch_size, 
        device=cfg.device
    )

    # create validation/ evaluation metrics
    evaluator = EvalModule(
        eval_cfgs = cfg.evaluation,
        body_model_type = cfg.body_model.type,
    ).to(cfg.device)

    # create optimizer
    optimizer = build_optimizer(
        cfg=cfg.model.regressor.optimizer,
        optimizer_type=cfg.model.regressor.optimizer.type,
        params=filter(lambda p: p.requires_grad, regressor.parameters()) 
    )

    # create renderer to visualize results
    renderer_camera = build_camera(
        camera_cfg=cfg.camera,
        camera_type=cfg.camera.type,
        batch_size=1,
        device=cfg.device
    ).to(cfg.device)
    renderer = Pytorch3dRenderer(
        cameras = renderer_camera.cameras,
        image_width=180,
        image_height=256,
    )

    # create losses
    criterion = LossModule(
        losses_cfgs = cfg.model.regressor.losses,
        body_model_type = cfg.body_model.type,
    ).to(cfg.device)

    # create optimizer module
    hhc_s2s_transformer = TrainModule(
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        diffusion=diffusion,
        model=regressor,
        criterion=criterion,
        evaluator=evaluator,
        body_model=body_model,
        renderer=renderer,
    ).to(cfg.device)
    
    # train model
    trainer = Trainer(
        train_cfg=cfg.training,
        train_module=hhc_s2s_transformer,
        optimizer=optimizer,
        logger=logger,
        device=cfg.device,
        batch_size=cfg.batch_size
    ).train()

if __name__ == "__main__":

    cfg = parse_args()
    if cfg.training.train or cfg.evluation.eval_val:
        train(cfg)