# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import os
import numpy as np
from loguru import logger as guru
from llib.optimizer.build import build_optimizer
from llib.training.fitter import Stopper

class HHCSOpti(nn.Module):
    """HHC optimizes two meshes using image keypoints and discrete
    contact labels."""
    def __init__(self,
                 opti_cfg,
                 camera,
                 body_model_h1,
                 body_model_h2,
                 criterion,
                 batch_size=1,
                 device='cuda',
                 diffusion_module=None,
    ):
        super(HHCSOpti, self).__init__()

        # save config file
        self.opti_cfg = opti_cfg
        self.batch_size = batch_size
        self.device = device
        self.print_loss = opti_cfg.print_loss

        self.camera = camera

        self.body_model_h1 = body_model_h1
        self.body_model_h2 = body_model_h2

        self.diffusion_module = diffusion_module

        # human optimization
        self.criterion = criterion
        self.num_iters = opti_cfg.hhcs.max_iters

        # parameters to be optimized 
        self.optimizables = {
            0: [
                'body_model_h1.translz',
                'body_model_h1.translx',
                'body_model_h1.transly',
                'body_model_h1.betas',
                'body_model_h1.body_pose',
                'body_model_h2.translx',
                'body_model_h2.transly',
                'body_model_h2.translz',
                'body_model_h2.betas',
                'body_model_h2.body_pose',
            ],    
            1: [
                'body_model_h1.translz',
                'body_model_h1.translx',
                'body_model_h1.transly',
                'body_model_h1.body_pose',
                'body_model_h2.translx',
                'body_model_h2.transly',
                'body_model_h2.translz',
                'body_model_h2.body_pose',
            ],
        }       
        

        # stop criterion 
        self.stopper = Stopper(
            num_prev_steps=opti_cfg.hhcs.num_prev_steps,
            slope_tol=opti_cfg.hhcs.slope_tol,
        )

    def setup_optimiables(self, stage):
        
        self.final_params = [] 

        optimizer_type = self.opti_cfg.optimizer.type
        lr = stage_lr = eval(f'self.opti_cfg.optimizer.{optimizer_type}.lr')
        if stage in [1]:
            stage_lr = lr / 10

        # camera parameters
        for param_name, param in self.named_parameters():
            if param_name in self.optimizables[stage]:
                param.requires_grad = True
                self.final_params.append({'params': param, 'lr': stage_lr})
            else:
                param.requires_grad = False
            
    @torch.no_grad()
    def fill_params(self, init_h1, init_h2, init_cam):
        device = self.body_model_h1.betas.device
        for param_name, param in self.body_model_h1.named_parameters():
            if param_name in init_h1.keys():
                init_value = init_h1[param_name].clone().detach().unsqueeze(0).to(device).requires_grad_(True)
                param[:] = init_value

        for param_name, param in self.body_model_h2.named_parameters():
            if param_name in init_h2.keys():
                init_value = init_h2[param_name].clone().detach().unsqueeze(0).to(device).requires_grad_(True)
                param[:] = init_value

        for param_name, param in self.camera.named_parameters():
            if param_name in init_cam.keys():
                init_value = init_cam[param_name].clone().detach().unsqueeze(0).to(device).requires_grad_(True)
                param[:] = init_value

        self.camera.iw[:] = init_cam['iw']
        self.camera.ih[:] = init_cam['ih']        


    def setup_optimizer(self, init_h1, init_h2, init_cam, stage):
        """Setup the optimizer for the current stage / reset in stages > 0."""

        # in the first stage, set the SMPL-X parameters to the initial values        
        if stage == 0:
            self.fill_params(init_h1, init_h2, init_cam)

        # pick the parameters to be optimized
        self.setup_optimiables(stage)

        # build optimizer
        self.optimizer = build_optimizer(
            self.opti_cfg.optimizer, 
            self.opti_cfg.optimizer.type,
            self.final_params
        )

    def print_losses(self, ld, stage=0, step=0, abbr=True):
        """Print the losses for the current stage."""
        total_loss = ld['total_loss'].item()
        out = f'Stage/step:{stage:2d}/{step:2} || Tl: {total_loss:.4f} || '
        for k, v in ld.items():
            if k != 'total_loss':
                kprint = ''.join([x[0] for x in k.split('_')]) if abbr else k
                if type(v) == torch.Tensor:
                    v = v.item()
                    out += f'{kprint}: {v:.4f} | '
        print(out)


    def optimize_humans(
        self,
        init_h1, 
        init_h2, 
        init_camera,
        contact_map,
        stage
    ):  
        """Optimize the human parameters for the given stage."""

        # set the loss weights for the current stage
        self.criterion.set_weights(stage)

        for i in range(self.num_iters[stage]):

            smpl_output_h1 = self.body_model_h1()
            smpl_output_h2 = self.body_model_h2()
            camera = self.camera

            # we tried different approaches / noies levels when using the SDS loss
            if self.opti_cfg.use_diffusion:
                if self.opti_cfg.sds_type == "fixed":
                    # use fixed value for noise level t
                    t_i = self.opti_cfg.sds_t_fixed
                elif self.opti_cfg.sds_type == "range":
                    # sample random integer between range lower and upper bound
                    t_min, t_max = self.opti_cfg.sds_t_range
                    t_i = np.random.randint(t_min, t_max, 1)[0]
                elif self.opti_cfg.sds_type == "adaptive":
                    # change noise level based on iteration
                    p = (self.num_iters[stage] - (i+1)) / self.num_iters[stage]
                    pidx = int(np.where(np.array(self.opti_cfg.sds_t_adaptive_i) > p)[0][-1])
                    t_i = self.opti_cfg.sds_t_adaptive_t[pidx]
            else:
                # without SDS loss, set t to None
                t_i = None

            # compute all loss
            loss, loss_dict = self.criterion(
                smpl_output_h1, 
                smpl_output_h2, 
                camera,
                init_h1, 
                init_h2,
                init_camera,
                contact_map,
                use_diffusion_prior=self.opti_cfg.use_diffusion,
                diffusion_module=self.diffusion_module,
                t=t_i,
            )

            if self.print_loss:
                self.print_losses(loss_dict, stage, i)

            # optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # break if stopping criterion is met
            stop_crit = self.stopper.check(loss.item())
            if stop_crit:
                break


    def fit(
        self, 
        init_h1, 
        init_h2,
        init_camera,
        contact_map,
    ): 
        """Main fitting function running through all stages of optimization""" 

        # we project the initial mesh to the image plane and use the keypoints 
        # if they're not visible in the image
        with torch.no_grad():
            self.fill_params(init_h1, init_h2, init_camera)
            init_h1['init_keypoints'] = self.camera.project(self.body_model_h1().joints)
            init_h2['init_keypoints'] = self.camera.project(self.body_model_h2().joints)

        # optimize in multiple stages
        for stage, _ in enumerate(range(len(self.num_iters))):
            guru.info(f'Starting with stage: {stage} \n')

            self.stopper.reset() # stopping criterion
            self.setup_optimizer(init_h1, init_h2, init_camera, stage) # setup optimizer

            # clone the initial estimate and detach it from the graph since it'll be used
            # as initialization and as prior the optimization
            if stage > 0:
                init_h1['body_pose'] = self.body_model_h1.body_pose.detach().clone()
                init_h2['body_pose'] = self.body_model_h2.body_pose.detach().clone()
                init_h1['betas'] = self.body_model_h1.betas.detach().clone()
                init_h2['betas'] = self.body_model_h2.betas.detach().clone()
            
            # run optmization for one stage
            self.optimize_humans(init_h1, init_h2, init_camera, contact_map, stage)
                
        # Get final loss value and get full skinning
        with torch.no_grad():
            smpl_output_h1 = self.body_model_h1()
            smpl_output_h2 = self.body_model_h2()

        return smpl_output_h1, smpl_output_h2