import torch.nn as nn
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from llib.utils.threed.conversion import axis_angle_to_rotation6d

from llib.losses.l2 import L2Loss
from llib.losses.build import build_loss
from llib.losses.contact import ContactMapLoss
from llib.utils.keypoints.gmfo import GMoF

class HHCOptiLoss(nn.Module):
    def __init__(
        self,
        losses_cfgs,
        body_model_type='smplx',
    ):
        super(HHCOptiLoss, self).__init__()

        self.cfg = losses_cfgs

        # when loss weigts are != 0, add loss as member variable
        for name, cfg in losses_cfgs.items():
            if name == 'debug':
                continue

            # add loss weight as member variable
            weight = cfg.weight
            setattr(self, name + '_weights', weight)
            
            # add criterion as member variable when weight != 0 exists
            if sum([x != 0 for x in cfg.weight]) > 0:

                function = build_loss(cfg, body_model_type)
                setattr(self, name + '_crit', function)

                # check if the criterion / loss is used in forward pass
                method = 'get_' + name + '_loss'
                assert callable(getattr(self, method)), \
                    f'Method {method} not implemented in HHCOptiLoss'

        self.set_weights(stage=0) # init weights with first stage

        self.robustifier = GMoF(rho=100.0)

        self.debug = []

    def set_weights(self, stage, default_stage=-1):

        for name, cfg in self.cfg.items():
            if name == 'debug':
                continue

            weight = getattr(self, name + '_weights')

            # use default stage value if weight for stage not specified
            weight_stage = default_stage if len(weight) <= stage else stage

            setattr(self, name + '_weight', weight[weight_stage])

    def get_keypoint2d_loss(self, vitpose, openpose, init_bev, est_joints, bs, num_joints, device):
        """Some keypoint processing to merge OpenPose and ViTPose keypoints."""
        
        gt_keypoints = vitpose #init['vitpose_keypoints'].unsqueeze(0).to(device)
        op_keypoints = openpose #init['op_keypoints'].unsqueeze(0).to(device)
        bs, nk, _ = gt_keypoints.shape

        # add openpose foot tip (missing in vitpose)
        ankle_joint = [11, 14]
        """
        ankle_thres = 5.0
        right_ankle_residual = torch.sum((gt_keypoints[:,11,:] - op_keypoints[:,11,:])**2)
        if right_ankle_residual < ankle_thres:
            gt_keypoints[:,22,:] = op_keypoints[:,22,:]
        left_ankle_residual = torch.sum((gt_keypoints[:,14,:] - op_keypoints[:,14,:])**2)
        if left_ankle_residual < ankle_thres:
            gt_keypoints[:,19,:] = op_keypoints[:,19,:]
        """

        # use initial (BEV) keypoints if detected ankle joints are missing/low confidence (e.g. when image is cropped)
        mask_init = (gt_keypoints < .2)[0,:,2]
        init_bev = init_bev # init['init_keypoints'] 
        init_keypoints = torch.cat([init_bev.double(), 0.5 * torch.ones(bs, nk, 1).to(device)], dim=-1)
        if mask_init[ankle_joint[0]] == 1:
            gt_keypoints[:,ankle_joint[0],:] = init_keypoints[:,ankle_joint[0],:]
            gt_keypoints[:,22:25,:] = init_keypoints[:,22:25,:]
        if mask_init[ankle_joint[1]] == 1:
            gt_keypoints[:,ankle_joint[1],:] = init_keypoints[:,ankle_joint[1],:]
            gt_keypoints[:,19:22,:] = init_keypoints[:,19:22,:]

        if gt_keypoints.shape[-1] == 3:
            gt_keypoints_conf = gt_keypoints[:, :, 2]
            gt_keypoints_vals = gt_keypoints[:, :, :2]
        else:
            gt_keypoints_vals = gt_keypoints
            gt_keypoints_conf = torch.ones([bs, num_joints], device=device)
        
        # normalize keypoint loss by bbox size 
        valid_kpts = gt_keypoints_vals[gt_keypoints_conf > 0]
        xmin, ymin = valid_kpts.min(0)[0]
        xmax, ymax = valid_kpts.max(0)[0]
        bbox_size = max(ymax-ymin, xmax-xmin)
        gt_keypoints_vals = gt_keypoints_vals / bbox_size * 512
        est_joints = est_joints / bbox_size * 512

        # robistify keypoints
        #residual = (gt_keypoints_vals - projected_joints) ** 2
        #rho = 100 ** 2
        #robust_residual = gt_keypoints_conf.unsqueeze(-1) * rho * \
        #                torch.div(residual, residual + rho)
        #keypoint2d_loss = torch.mean(robust_residual) * self.keypoint2d_weight

        # comput keypoint loss
        keypoint2d_loss = self.keypoint2d_crit(
            gt_keypoints_vals, est_joints, gt_keypoints_conf
        ) * self.keypoint2d_weight

        return keypoint2d_loss

    def get_shape_prior_loss(self, betas):
        shape_prior_loss = self.shape_prior_crit(
            betas, y=None) * self.shape_prior_weight
        return shape_prior_loss

    def get_pose_prior_loss(self, pose):
        pose_prior_loss = torch.sum(self.pose_prior_crit(
            pose)) * self.pose_prior_weight
        return pose_prior_loss

    def get_init_pose_loss(self, init_pose, est_body_pose, device):
        
        if len(init_pose.shape) == 1:
            init_pose = init_pose.unsqueeze(0)
        
        init_pose_prior_loss = self.init_pose_crit(
            init_pose, est_body_pose
        ) * self.init_pose_weight

        return init_pose_prior_loss

    def get_init_shape_loss(self, init_shape, est_shape, device):
        init_shape_loss = self.init_pose_crit(
            init_shape, est_shape
            ) * self.init_shape_weight
        return init_shape_loss

    def get_init_transl_loss(self, init_transl, est_transl, device):
        init_transl_loss = self.init_pose_crit(
            init_transl, est_transl
            ) * self.init_transl_weight
        return init_transl_loss
    
    def get_hhc_contact_loss(self, contact_map, vertices_h1, vertices_h2):
       
        hhc_contact_loss = self.hhc_contact_crit(
                    v1=vertices_h1, 
                    v2=vertices_h2, 
                    cmap=contact_map, 
                    factor=100
        ) * self.hhc_contact_weight

        return hhc_contact_loss

    def get_hhc_contact_general_loss(self, vertices_h1, vertices_h2):
        hhc_contact_general_loss = self.hhc_contact_general_crit(
                    v1=vertices_h1, 
                    v2=vertices_h2,
                    factor=100
        ) * self.hhc_contact_general_weight
        return hhc_contact_general_loss

    def get_ground_plane_loss(self, vertices):
        raise NotImplementedError

    def get_diffusion_prior_orient_loss(self, global_orient_diffused, global_orient_current):
        global_orient_loss = self.diffusion_prior_global_crit(
            global_orient_diffused, global_orient_current) * \
                self.diffusion_prior_global_weight
        return global_orient_loss

    def get_diffusion_prior_pose_loss(self, body_pose_diffused, body_pose_current):
        body_pose_loss = self.diffusion_prior_body_crit(
            body_pose_diffused, body_pose_current) * \
                self.diffusion_prior_body_weight
        return body_pose_loss
    
    def get_diffusion_prior_shape_loss(self, betas_diffused, betas_current):
        betas_loss = self.diffusion_prior_shape_crit(
            betas_diffused, betas_current) * \
                self.diffusion_prior_shape_weight
        return betas_loss
    
    def get_diffusion_prior_scale_loss(self, betas_diffused, betas_current):
        betas_loss = self.diffusion_prior_shape_crit(
            betas_diffused, betas_current) * \
                self.diffusion_prior_shape_weight
        return betas_loss

    def get_diffusion_prior_transl_loss(self, transl_diffused, transl_current):
        transl_loss = self.diffusion_prior_transl_crit(
            transl_diffused, transl_current) * \
                self.diffusion_prior_transl_weight
        return transl_loss

    def undo_orient_and_transl(self, diffusion_module, x_start_smpls, target_rotation, target_transl):
        
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

    def forward_diffusion(
        self,
        diffusion_module, # the diffusion module
        t, # noise level
        smpl_output_h1, # the current estimate of person a
        smpl_output_h2, # the current estimate of person b
        guidance_params={}, # the initial estimate of person a and b
    ):
        """The SDS loss or L_diffusion as we define it in the paper"""

        ld = {} # store losses in dict for printing
        device = diffusion_module.cfg.device

        # take the current estimate of the optimization
        x_start_smpls = [smpl_output_h1, smpl_output_h2]

        # fix because optimization data loader does not do the flipping anymore
        #if smpl_output_h1.transl[0,0] > smpl_output_h2.transl[0,0]:
        #    x_start_smpls = [smpl_output_h2, smpl_output_h1]

        dbs = diffusion_module.bs

        # Run a diffuse-denoise step. To do this, we use torch.no_grad() to
        # ensure that the gradients are not propagated through the diffusion
        with torch.no_grad():
            # first, we need to transform the current estimate of the optimization to 
            # BUDDI's format
            init_rotation =  axis_angle_to_matrix(x_start_smpls[0].global_orient).detach().clone().repeat(dbs, 1, 1)
            init_transl =  x_start_smpls[0].transl.detach().clone().repeat(dbs, 1, 1)
            x = {
                'orient': torch.cat([
                    axis_angle_to_rotation6d(x_start_smpls[0].global_orient.unsqueeze(1)), 
                    axis_angle_to_rotation6d(x_start_smpls[1].global_orient.unsqueeze(1))], dim=1).repeat(dbs, 1, 1),
                'pose': torch.cat([
                    axis_angle_to_rotation6d(x_start_smpls[0].body_pose.unsqueeze(1).view(1, 1, -1, 3)).view(1, 1, -1), 
                    axis_angle_to_rotation6d(x_start_smpls[1].body_pose.unsqueeze(1).view(1, 1, -1, 3)).view(1, 1, -1)], dim=1).repeat(dbs, 1, 1),
                'shape': torch.cat([
                    torch.cat((x_start_smpls[0].betas, x_start_smpls[0].scale), dim=-1).unsqueeze(1),
                    torch.cat((x_start_smpls[1].betas, x_start_smpls[1].scale), dim=-1).unsqueeze(1)], dim=1).repeat(dbs, 1, 1),
                'transl': torch.cat([
                    x_start_smpls[0].transl.unsqueeze(1), 
                    x_start_smpls[1].transl.unsqueeze(1)], dim=1).repeat(dbs, 1, 1)
            }

            # if len(diffusion_module.exp_cfg.guidance_params) > 0:
            #     guidance_params = {
            #         'orient': init_human['global_orient'].unsqueeze(0).repeat(dbs, 1, 1),
            #         'pose': init_human['body_pose'].unsqueeze(0).repeat(dbs, 1, 1),
            #         'shape': torch.cat((init_human['betas'], init_human['scale'].unsqueeze(1)), dim=-1).unsqueeze(0).repeat(dbs, 1, 1),
            #         'transl': init_human['transl'].unsqueeze(0).repeat(dbs, 1, 1)
            #     }
            #     guidance_params = diffusion_module.cast_smpl(guidance_params)
            #     guidance_params = diffusion_module.split_humans(guidance_params)
            # else:
            #     guidance_params = {} # no guidance params are used here
            x = diffusion_module.reset_orient_and_transl(x) # use relative translation

            # run the diffusion (diffuse parameters and use BUDDI to denoise them)
            t = torch.tensor([t] * diffusion_module.bs).to(diffusion_module.cfg.device)
            diffusion_output = diffusion_module.diffuse_denoise(x=x, y=guidance_params, t=t)
            denoised_smpls = diffusion_output['denoised_smpls'] 

            # now we need to bring the estimates back into the format of the original optimization
            new_orient, new_transl = self.undo_orient_and_transl(
                diffusion_module, denoised_smpls, init_rotation, init_transl)
            x_end_smpls = diffusion_module.get_smpl({
                'orient': axis_angle_to_rotation6d(new_orient).repeat(dbs, 1, 1),
                'pose': torch.cat([
                    axis_angle_to_rotation6d(denoised_smpls[0].body_pose.view(dbs,-1,3)).unsqueeze(1), 
                    axis_angle_to_rotation6d(denoised_smpls[1].body_pose.view(dbs,-1,3)).unsqueeze(1)], dim=1), 
                'shape': torch.cat([
                    torch.cat((denoised_smpls[0].betas, denoised_smpls[0].scale), dim=-1).unsqueeze(1),
                    torch.cat((denoised_smpls[1].betas, denoised_smpls[1].scale), dim=-1).unsqueeze(1)], dim=1), 
                'transl': new_transl.repeat(dbs, 1, 1)
            })

        # Regularize the orientation and pose of the two people
        if self.diffusion_prior_orient_weight > 0:
            ld['regularize_h_0_orient'] = self.diffusion_prior_orient_weight * \
                torch.norm(x_start_smpls[0].global_orient[[0]] - x_end_smpls[0].global_orient[[0]].detach())
            ld['regularize_h_1_orient'] = self.diffusion_prior_orient_weight * \
                torch.norm(x_start_smpls[1].global_orient[[0]] - x_end_smpls[1].global_orient[[0]].detach())

        # Regularize the pose of the two people
        if self.diffusion_prior_pose_weight > 0:
            ld['regularize_h_0_pose'] = self.diffusion_prior_pose_weight * \
                    torch.norm(x_start_smpls[0].body_pose[[0]] - x_end_smpls[0].body_pose[[0]].detach())
            ld['regularize_h_1_pose'] = self.diffusion_prior_pose_weight * \
                torch.norm(x_start_smpls[1].body_pose[[0]] - x_end_smpls[1].body_pose[[0]].detach())
        
        # Regularize the relative translation of the two people
        if self.diffusion_prior_transl_weight > 0:
            # t1 - t0 = t2 - t1
            diffusion_dist = x_end_smpls[1].transl[[0]].detach() - x_end_smpls[0].transl[[0]].detach()
            curr_dist = x_start_smpls[1].transl[[0]] - x_start_smpls[0].transl[[0]]
            ld['regularize_h_1_h_0_transl'] = self.diffusion_prior_transl_weight * \
                                                                torch.norm(diffusion_dist - curr_dist)

        # Regularize the shape of the two people
        if self.diffusion_prior_shape_weight > 0:
            ld['regularize_h_0_shape'] = self.diffusion_prior_shape_weight * \
                torch.norm(x_start_smpls[0].betas[[0]] - x_end_smpls[0].betas[[0]].detach())
            ld['regularize_h_1_shape'] = self.diffusion_prior_shape_weight * \
            torch.norm(x_start_smpls[1].betas[[0]] - x_end_smpls[1].betas[[0]].detach())

        # Regularize the scale of the two people 
        if self.diffusion_prior_scale_weight > 0:
            ld['regularize_h_0_scale'] = self.diffusion_prior_scale_weight * \
                torch.norm(x_start_smpls[0].scale[[0]] - x_end_smpls[0].scale[[0]].detach())
            ld['regularize_h_1_scale'] = self.diffusion_prior_scale_weight * \
                torch.norm(x_start_smpls[1].scale[[0]] - x_end_smpls[1].scale[[0]].detach())
    
        # Sum the loss terms
        diffusion_loss = torch.tensor([0.0]).to(device)
        for k, v in ld.items():
            diffusion_loss += v 

        # average the losses over batch
        ld_out = {}
        for k, v in ld.items():
            if type(v) == torch.Tensor:
                ld_out[k] = v.mean()

        # final loss value
        diffusion_loss = sum(ld_out.values())
        ld_out['total_sds_loss'] = diffusion_loss

        return diffusion_loss, ld_out


    def forward_fitting(
        self, 
        smpl_output_h1, # the current estimate of person a
        smpl_output_h2, # the current estimate of person b
        camera, # camera
        #init_h1, # the initial estimate of person a (from BEV) 
        #init_h2, # the initial estimate of person b (from BEV)
        init_human, # the initial estimate of person a and b 
        init_camera, # BEV camera
        contact_map, # the contact map between the two people
    ):  
        bs, num_joints, _ = smpl_output_h1.joints.shape
        device = smpl_output_h1.joints.device

        ld = {} # store losses in dict for printing

        #init_h1_betas = init_h1['betas'].unsqueeze(0).to(device)
        #init_h2_betas = init_h2['betas'].unsqueeze(0).to(device)

        # project 3D joinst to 2D
        projected_joints_h1 = camera.project(smpl_output_h1.joints)
        projected_joints_h2 = camera.project(smpl_output_h2.joints)

        # keypoint losses for each human
        ld['keypoint2d_losses'] = 0.0
        if self.keypoint2d_weight > 0:
            ld['keypoint2d_losses'] += self.get_keypoint2d_loss(
                init_human['keypoints'][[0]],
                init_human['op_keypoints'][[0]], 
                init_human['init_keypoints'][[0]],  
                projected_joints_h1, bs, num_joints, device)
            ld['keypoint2d_losses'] += self.get_keypoint2d_loss(
                init_human['keypoints'][[1]],
                init_human['op_keypoints'][[1]], 
                init_human['init_keypoints'][[1]],  
                projected_joints_h2, bs, num_joints, device)

        # shape prior loss
        ld['shape_prior_loss'] = 0.0
        if self.shape_prior_weight > 0:
            ld['shape_prior_loss'] += self.get_shape_prior_loss(
                smpl_output_h1.betas)
            ld['shape_prior_loss'] += self.get_shape_prior_loss(
                smpl_output_h2.betas)
        
        # pose prior loss
        ld['pose_prior_loss'] = 0.0
        if self.pose_prior_weight > 0:
            ld['pose_prior_loss'] += self.get_pose_prior_loss(
                smpl_output_h1.body_pose)
            ld['pose_prior_loss'] += self.get_pose_prior_loss(
                smpl_output_h2.body_pose)

        # pose prior losses for each human
        ld['init_pose_losses'] = 0.0
        if self.init_pose_weight > 0:
            ld['init_pose_losses'] += self.get_init_pose_loss(
                init_human['body_pose'][[0]], smpl_output_h1.body_pose, device)
            ld['init_pose_losses'] += self.get_init_pose_loss(
                init_human['body_pose'][[1]], smpl_output_h2.body_pose, device)

        # shape prior losses for each human
        ld['init_shape_losses'] = 0.0
        if self.init_shape_weight > 0:
            ld['init_shape_losses'] += self.get_init_shape_loss(
                init_human['betas'][[0]], smpl_output_h1.betas, device)
            ld['init_shape_losses'] += self.get_init_shape_loss(
                init_human['betas'][[1]], smpl_output_h2.betas, device)

        # shape prior losses for each human
        ld['init_transl_losses'] = 0.0
        if self.init_transl_weight > 0:
            ld['init_transl_losses'] += self.get_init_transl_loss(
                init_human['transl'][[0]], smpl_output_h1.transl, device)
            ld['init_transl_losses'] += self.get_init_transl_loss(
                init_human['transl'][[1]], smpl_output_h2.transl, device)
        
        # contact loss between two humans
        ld['hhc_contact_loss'] = 0.0
        if self.hhc_contact_weight:
            ld['hhc_contact_loss'] += self.get_hhc_contact_loss(
                contact_map, smpl_output_h1.vertices, smpl_output_h2.vertices)

        # contact loss between two humans
        ld['hhc_contact_general_loss'] = 0.0
        if self.hhc_contact_general_weight:
            ld['hhc_contact_general_loss'] += self.get_hhc_contact_general_loss(
                smpl_output_h1.vertices, smpl_output_h2.vertices)

        # ground plane assumption loss
        ld['ground_plane_loss'] = 0.0
        if self.ground_plane_weight:
            ld['ground_plane_loss'] += self.get_ground_plane_loss(smpl_output_h1.vertices)
            ld['ground_plane_loss'] += self.get_ground_plane_loss(smpl_output_h2.vertices) 

        # average the losses over batch
        ld_out = {}
        for k, v in ld.items():
            if type(v) == torch.Tensor:
                ld_out[k] = v.mean()

        # final loss value
        fitting_loss = sum(ld_out.values())
        ld_out['total_fitting_loss'] = fitting_loss

        return fitting_loss, ld_out


    def forward(
        self, 
        smpl_output_h1, 
        smpl_output_h2,
        camera,
        #init_h1, 
        #init_h2,
        init_human,
        init_camera,
        contact_map,
        use_diffusion_prior=False,
        diffusion_module=None,
        t=None,
        guidance_params={},
    ): 
        """
        Compute all losses in the current optimization iteration.
        The current estimate is smpl_output_h1/smpl_output_h2, which
        we pass to the L_fitting and L_diffusion modules. The final
        loss is the sum of both losses.
        """ 
        
        # fitting losses (keypoints, pose / shape prior etc.)
        fitting_loss, fitting_ld_out = self.forward_fitting(
            smpl_output_h1, 
            smpl_output_h2,
            camera,
            #init_h1, 
            #init_h2,
            init_human,
            init_camera,
            contact_map
        )

        # diffusion prior loss / sds loss / BUDDI loss
        if use_diffusion_prior:
            sds_loss, sds_ld_out = self.forward_diffusion(
                diffusion_module,
                t,
                smpl_output_h1, 
                smpl_output_h2,
                guidance_params, # for bev conditioning
            )

        # update loss dict and sum up losses
        if use_diffusion_prior:
            total_loss = fitting_loss + sds_loss
            ld_out = {**fitting_ld_out, **sds_ld_out}
        else:
            total_loss = fitting_loss
            ld_out = fitting_ld_out
        
        ld_out['total_loss'] = total_loss

        return total_loss, ld_out