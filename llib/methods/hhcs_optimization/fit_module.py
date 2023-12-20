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
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

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
                 renderer=None
    ):
        super(HHCSOpti, self).__init__()

        # save config file
        self.opti_cfg = opti_cfg
        self.batch_size = batch_size
        self.device = device
        self.print_loss = opti_cfg.print_loss
        self.render_iters = opti_cfg.render_iters

        self.camera = camera

        self.body_model_h1 = body_model_h1
        self.body_model_h2 = body_model_h2
        self.faces = torch.from_numpy(
            body_model_h1.faces.astype(np.int32)).to(self.device)

        self.diffusion_module = diffusion_module

        # human optimization
        self.criterion = criterion
        self.num_iters = opti_cfg.hhcs.max_iters

        # parameters to be optimized
        self.optimizables = {
            0: [
                'body_model_h1.transl',
                'body_model_h1.betas',
                'body_model_h1.body_pose',
                # 'body_model_h1.global_orient',
                'body_model_h2.transl',
                'body_model_h2.betas',
                'body_model_h2.body_pose',
                # 'body_model_h2.global_orient',
            ],    
            1: [
                'body_model_h1.transl',
                'body_model_h1.body_pose',
                'body_model_h2.transl',
                'body_model_h2.body_pose',
            ],
        }       

        # if bev guidance also optimize the body global orientation 
        # if len(self.diffusion_module.exp_cfg.guidance_params) > 0:
            # self.optimizables[0].extend([
                # 'body_model_h1.global_orient',
                # 'body_model_h2.global_orient',
            # ])
        

        # stop criterion 
        self.stopper = Stopper(
            num_prev_steps=opti_cfg.hhcs.num_prev_steps,
            slope_tol=opti_cfg.hhcs.slope_tol,
        )

        # rendered images per iter
        if self.render_iters:
            self.renderer = renderer
            self.renderings = []

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
    def fill_params(self, init_human, init_cam):
        """Fill the parameters of the human model and camera with the
        initial values."""

        device = self.body_model_h1.betas.device
        for param_name, param in self.body_model_h1.named_parameters():
            if param_name in init_human.keys():
                init_value = init_human[param_name][[0]].clone().detach().to(device).requires_grad_(True)
                param[:] = init_value

        for param_name, param in self.body_model_h2.named_parameters():
            if param_name in init_human.keys():
                init_value = init_human[param_name][[1]].clone().detach().to(device).requires_grad_(True)
                param[:] = init_value

        for param_name, param in self.camera.named_parameters():
            if param_name in init_cam.keys():
                init_value = init_cam[param_name].clone().detach().unsqueeze(0).to(device).requires_grad_(True)
                param[:] = init_value

        self.camera.iw[:] = init_cam['iw']
        self.camera.ih[:] = init_cam['ih']        


    def setup_optimizer(self, init_human, init_cam, stage):
        """Setup the optimizer for the current stage / reset in stages > 0."""

        # in the first stage, set the SMPL-X parameters to the initial values        
        if stage == 0:
            self.fill_params(init_human, init_cam)

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
    
    def render_current_estimate(self, stage="", iter="", color=['light_blue3', 'light_blue5']):
        """Render the current estimates"""

        v1 = self.body_model_h1().vertices.detach()
        v2 = self.body_model_h2().vertices.detach()
        verts = torch.cat([v1,v2], dim=0)

        bm = 'smpl' if verts.shape[1] == 6890 else 'smplx'
        self.renderer.update_camera_pose(
            self.camera.pitch.item(), self.camera.yaw.item(), self.camera.roll.item(), 
            self.camera.tx.item(), self.camera.ty.item(), self.camera.tz.item()
        )
        rendered_img = self.renderer.render(verts, self.faces, colors = color, body_model=bm)
        color_image = rendered_img[0].detach().cpu().numpy() * 255
        self.renderings.append(color_image)


    def optimize_humans(
        self,
        #init_h1, 
        #init_h2, 
        init_human,
        init_camera,
        contact_map,
        stage,
        guidance_params={},
    ):  
        """Optimize the human parameters for the given stage."""

        # set the loss weights for the current stage
        self.criterion.set_weights(stage)

        for i in range(self.num_iters[stage]):
            
            if self.render_iters:
                colors = {0: ['paper_blue', 'paper_red'], 1: ['paper_blue', 'paper_red']}
                self.render_current_estimate(stage, i, colors[stage])

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
                #init_h1, 
                #init_h2,
                init_human,
                init_camera,
                contact_map,
                use_diffusion_prior=self.opti_cfg.use_diffusion,
                diffusion_module=self.diffusion_module,
                t=t_i,
                guidance_params=guidance_params,
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
        #init_h1, 
        #init_h2,
        init_human,
        init_camera,
        contact_map,
    ): 
        """Main fitting function running through all stages of optimization"""

        # we project the initial mesh to the image plane and use the keypoints 
        # if they're not visible in the image
        with torch.no_grad():
            self.fill_params(init_human, init_camera)
            init_human['init_keypoints'] = torch.cat([
                self.camera.project(self.body_model_h1().joints),
                self.camera.project(self.body_model_h2().joints)], axis=0)

        # copy init human params for guidance
        #guidance_params = {k: v.clone().detach() for k, v in init_human.items()}
        guidance_params = {}
        if self.diffusion_module is not None:
            if len(self.diffusion_module.exp_cfg.guidance_params) > 0:
                dbs = self.diffusion_module.bs
                guidance_params = {
                    'orient': init_human['global_orient'].unsqueeze(0).repeat(dbs, 1, 1),
                    'pose': init_human['body_pose'].unsqueeze(0).repeat(dbs, 1, 1),
                    'shape': torch.cat((init_human['betas'], init_human['scale'].unsqueeze(1)), dim=-1).unsqueeze(0).repeat(dbs, 1, 1),
                    'transl': init_human['transl'].unsqueeze(0).repeat(dbs, 1, 1)
                }
                guidance_params = self.diffusion_module.cast_smpl(guidance_params)
                guidance_params = self.diffusion_module.split_humans(guidance_params)
        #else:
        #    guidance_params = {} # no guidance params are used here


        def undo_orient_and_transl(diffusion_module, x_start_smpls, target_rotation, target_transl):
            
            #orient, cam_rotation
            global_orient_h0 = x_start_smpls[0].global_orient.unsqueeze(1) #.repeat(64,1,1)
            global_orient_h1 = x_start_smpls[1].global_orient.unsqueeze(1) #.repeat(64,1,1)
            param = torch.cat((global_orient_h0, global_orient_h1), dim=1)
            param_rotmat = axis_angle_to_matrix(param)
            cam_rotation = torch.einsum('bml,bln->bmn', target_rotation, param_rotmat[:, 0, :, :].transpose(2, 1))
            new_orient = matrix_to_axis_angle(torch.einsum('bnm,bhml->bhnl', cam_rotation, param_rotmat))
            new_orient=new_orient[[0],:,:]

            pelvis = torch.cat((
                diffusion_module.body_model(betas=x_start_smpls[0].betas, scale=x_start_smpls[0].scale).joints[:,[0],:],
                diffusion_module.body_model(betas=x_start_smpls[1].betas, scale=x_start_smpls[1].scale).joints[:,[0],:]
            ), dim=1)

            transl_h0 = x_start_smpls[0].transl.unsqueeze(1) #.repeat(64,1,1)
            transl_h1 = x_start_smpls[1].transl.unsqueeze(1) #.repeat(64,1,1)
            transl = torch.cat((transl_h0, transl_h1), dim=1)
            root_transl = transl[:,[0],:]
            cam_translation = (-1 * torch.einsum('bhn,bnm->bhm', target_transl + pelvis, cam_rotation)) + root_transl + pelvis
            xx = transl + pelvis - cam_translation
            new_transl = torch.einsum('bhn,bnm->bhm', xx, cam_rotation.transpose(2, 1)) - pelvis
            new_transl=new_transl[[0],:,:]

            return new_orient, new_transl

        ############ conditional sampling ##############
        if len(guidance_params) > 0:
            # guru.info('Start sampling unconditional')
            cond_ts = np.arange(1, self.diffusion_module.diffusion.num_timesteps, 100)[::-1]
            log_freq = cond_ts.shape[0] # no logging
            x_ts, x_starts = self.diffusion_module.sample_from_model(
                cond_ts, log_freq, guidance_params
            )
            # undo orient and transl
            init_rotation =  axis_angle_to_matrix(init_human['global_orient'][0]).detach().clone().repeat(dbs, 1, 1)
            init_transl =  init_human['transl'][0].detach().clone().repeat(dbs, 1, 1)
            new_orient, new_transl = undo_orient_and_transl(
                self.diffusion_module, x_starts['final'], init_rotation, init_transl
            )
            for i in range(2):
                for param in ['global_orient', 'body_pose', 'betas', 'transl', 'scale']:
                    if param == 'global_orient':
                        init_human[param][i] = new_orient[0][i]
                    elif param == 'transl':
                        init_human[param][i] = new_transl[0][i]
                    else:
                        i_param = x_starts['final'][i]
                        init_human[param][i] = eval(f'i_param.{param}')[0].detach().clone()

            # we project the initial mesh to the image plane and use the keypoints 
            # if they're not visible in the image
            with torch.no_grad():
                self.fill_params(init_human, init_camera)
                init_human['init_keypoints'] = torch.cat([
                    self.camera.project(self.body_model_h1().joints),
                    self.camera.project(self.body_model_h2().joints)], axis=0) 

        # optimize in multiple stages
        for stage, _ in enumerate(range(len(self.num_iters))):
            guru.info(f'Starting with stage: {stage} \n')

            self.stopper.reset() # stopping criterion
            self.setup_optimizer(init_human, init_camera, stage) # setup optimizer

            # clone the initial estimate and detach it from the graph since it'll be used
            # as initialization and as prior the optimization
            if stage > 0:
                init_human['body_pose'] = torch.cat([
                    self.body_model_h1.body_pose.detach().clone(),
                    self.body_model_h2.body_pose.detach().clone()
                ], axis=0)
                init_human['betas'] = torch.cat([
                    self.body_model_h1.betas.detach().clone(),
                    self.body_model_h2.betas.detach().clone()
                ], axis=0)
            
            # run optmization for one stage
            self.optimize_humans(init_human, init_camera, contact_map, stage, guidance_params)
                
        # Get final loss value and get full skinning
        with torch.no_grad():
            smpl_output_h1 = self.body_model_h1()
            smpl_output_h2 = self.body_model_h2()

        return smpl_output_h1, smpl_output_h2
