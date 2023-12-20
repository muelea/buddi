import torch.nn as nn
import numpy as np
import torch
from llib.losses.l2 import L2Loss
from llib.losses.build import build_loss
from llib.losses.contact import ContactMapLoss
from llib.utils.threed.conversion import batch_rodrigues

class LossModule(nn.Module):
    def __init__(
        self,
        losses_cfgs,
        body_model_type='smplx'
    ):
        super(LossModule, self).__init__()

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

        # build contact loss criterion for heatmap computation
        function = build_loss(self.cfg.hhc_contact, body_model_type)
        setattr(self, 'hhc_contact_module', function)

    def set_weights(self, stage, default_stage=-1):

        for name, cfg in self.cfg.items():
            if name == 'debug':
                continue
            
            weight = getattr(self, name + '_weights')

            # use default stage value if weight for stage not specified
            if len(weight) <= stage: 
                stage = default_stage

            setattr(self, name + '_weight', weight[stage])

    def get_keypoint2d_loss(self, gt_keypoints, projected_joints, bs, num_joints, device):
        raise NotImplementedError

    def get_shape_prior_loss(self, betas):
        raise NotImplementedError

    def get_pose_prior_loss(self, pose, betas):
        raise NotImplementedError

    def get_init_pose_loss(self, init_pose, est_body_pose, device):
        raise NotImplementedError

    def get_pseudogt_pose_loss(self, init_pose, est_body_pose, device):
        """Pose prior loss (pushes to pseudo-ground truth pose)"""
        bs = init_pose.shape[0]
        init_pose_rotmat = batch_rodrigues(init_pose.view(bs, -1, 3).reshape(-1, 3)).view(bs, -1, 3, 3)
        est_body_pose_rotmat = batch_rodrigues(est_body_pose.reshape(bs, -1, 3).reshape(-1,3)).reshape(bs, -1, 3, 3)
        mseloss = nn.MSELoss().to('cuda')
        init_pose_prior_loss = (
            (init_pose_rotmat - est_body_pose_rotmat)**2
        ).sum((1,2,3)).mean() * self.pseudogt_pose_weight
        return init_pose_prior_loss

    def get_pseudogt_shape_loss(self, init_shape, est_shape, device):
        """Shape parameter loss (pushes to pseudo-ground truth shape)"""
        pgt_shape_loss = self.pseudogt_shape_crit(
            init_shape, est_shape) * self.pseudogt_shape_weight
        return pgt_shape_loss

    def get_pseudogt_v2v_loss(self, init_verts, est_verts, device):
        """3D v2v loss (pushes to pseudo-ground truth shape)"""
        pgt_v2v_loss = ((init_verts - est_verts)**2).sum(-1).mean(-1).mean() \
             * self.pseudogt_v2v_weight
        return pgt_v2v_loss

    def get_pseudogt_j2j_loss(self, init_joints, est_joints, device):
        """3D joint-to-joint loss (pushes to pseudo-ground truth shape)"""
        pgt_j2j_loss = ((init_joints - est_joints)**2).sum(-1).mean(-1).mean() \
             * self.pseudogt_v2v_weight
        return pgt_j2j_loss

    def get_pseudogt_transl_loss(self, init_transl, est_transl, device):
        """Translation loss (pushes to pseudo-ground truth translation)"""
        init_transl = init_transl
        est_transl = est_transl
        pgt_transl_loss = self.pseudogt_transl_crit(
            init_transl, est_transl) * self.pseudogt_transl_weight
        return pgt_transl_loss

    def get_hhc_contact_loss(self, contact_map, vertices_h1, vertices_h2): 
        loss = self.hhc_contact_crit(v1=vertices_h1, v2=vertices_h2, 
                cmap=contact_map, factor=100) 
        loss = loss.mean() * self.hhc_contact_weight
        return loss

    def get_cmap_loss(self, tar_contact_map_binary, est_contact_map): 
        raise NotImplementedError

    def get_cmap_heat_token_loss(self, tar_cmap_heat, est_cmap_token):
        raise NotImplementedError

    def get_cmap_heat_smpl_loss(self, tar_cmap_heat, est_vertices_h1, est_vertices_h2):
        raise NotImplementedError

    def get_cmap_binary_token_loss(self, tar_cmap_binary, est_cmap_token):
        raise NotImplementedError

    def get_cmap_binary_smpl_loss(self, tar_cmap_binary, est_vertices_h1, est_vertices_h2):
        raise NotImplementedError

    def get_hhc_contact_general_loss(self, vertices_h1, vertices_h2):
        raise NotImplementedError

    def get_ground_plane_loss(self, vertices):
        raise NotImplementedError

    def zero_loss_dict(self):
        ld = {} 
        ld['shape_prior_loss_0'] = 0.0
        ld['shape_prior_loss_1'] = 0.0
        ld['pose_prior_loss_0'] = 0.0
        ld['pose_prior_loss_1'] = 0.0
        ld['pseudogt_pose_losses_0'] = 0.0
        ld['pseudogt_pose_losses_1'] = 0.0
        ld['pseudogt_global_orient_losses_0'] = 0.0
        ld['pseudogt_global_orient_losses_1'] = 0.0
        ld['pseudogt_shape_losses_0'] = 0.0
        ld['pseudogt_shape_losses_1'] = 0.0
        ld['pseudogt_scale_losses_0'] = 0.0
        ld['pseudogt_scale_losses_1'] = 0.0
        ld['pseudogt_transl_losses_0'] = 0.0
        ld['pseudogt_transl_losses_1'] = 0.0
        ld['pseudogt_v2v_losses_0'] = 0.0
        ld['pseudogt_v2v_losses_1'] = 0.0
        ld['pseudogt_j2j_losses_0'] = 0.0
        ld['pseudogt_j2j_losses_1'] = 0.0
        ld['hhc_contact_loss'] = 0.0
        ld['cmap_loss'] = 0.0
        ld['cmap_heat_smpl_loss'] = 0.0
        ld['cmap_heat_token_loss'] = 0.0
        ld['cmap_binary_smpl_loss'] = 0.0
        ld['cmap_binary_token_loss'] = 0.0
        ld['hhc_contact_general_loss'] = 0.0
        return ld

    def forward(
        self, 
        est_smpl, # estimated smpl
        tar_smpl, # target smpl
        est_contact_map=None,
        tar_contact_map=None,
        tar_contact_map_binary=None,

    ):  

        bs, num_joints, _ = est_smpl[0].joints.shape
        device = est_smpl[0].joints.device

        ld = self.zero_loss_dict() # store losses in dict

        # contact loss between two humans
        if self.hhc_contact_weight:
            ld['hhc_contact_loss'] += self.get_hhc_contact_loss(
                tar_contact_map_binary, est_smpl[0].vertices, est_smpl[1].vertices)

        # predicted contact map loss
        if self.cmap_weight:
            ld['cmap_loss'] += self.get_cmap_loss(
                tar_contact_map_binary, est_contact_map)        

        # free contact loss between two humans (e.g. to resolve intersections)
        if self.hhc_contact_general_weight:
            ld['hhc_contact_general_loss'] += self.get_hhc_contact_general_loss(
                est_smpl[0].vertices, est_smpl[1].vertices)

        # ToDo: ADD NEW LOSSES
        if self.cmap_heat_token_weight:
            ld['cmap_heat_token_loss'] += self.get_cmap_heat_token_loss(
                tar_contact_map, est_contact_map)
        
        if self.cmap_heat_smpl_weight:
            ld['cmap_heat_smpl_loss'] += self.get_cmap_heat_smpl_loss(
                tar_contact_map, est_smpl[0].vertices, est_smpl[1].vertices)
        
        if self.cmap_binary_token_weight:
            ld['cmap_binary_token_loss'] += self.get_cmap_binary_token_loss(
                tar_contact_map_binary, est_contact_map)

        if self.cmap_binary_smpl_weight:
            ld['cmap_binary_smpl_loss'] += self.get_cmap_binary_smpl_loss(
                tar_contact_map_binary, est_smpl[0].vertices, est_smpl[1].vertices)



        # per human losses
        for hidx in range(len(est_smpl)):
            h = f'_{hidx}'

            # shape prior loss
            if self.shape_prior_weight > 0:
                ld['shape_prior_loss'+h] += self.get_shape_prior_loss(
                    est_smpl[hidx].betas)

            # pose prior loss
            if self.pose_prior_weight > 0:
                ld['pose_prior_loss'+h] += self.get_pose_prior_loss(
                    est_smpl[hidx].body_pose, est_smpl[hidx].betas)
           
            # pose prior losses for each human
            if self.pseudogt_pose_weight > 0:
                ld['pseudogt_pose_losses'+h] += self.get_pseudogt_pose_loss(
                    tar_smpl[hidx].body_pose, est_smpl[hidx].body_pose, device)
                ld['pseudogt_global_orient_losses'+h] += self.get_pseudogt_pose_loss(
                    tar_smpl[hidx].global_orient, est_smpl[hidx].global_orient, device)

            # pose prior losses for each human
            if self.pseudogt_shape_weight > 0:
                # concat scale and betas
                ld['pseudogt_shape_losses'+h] += self.get_pseudogt_shape_loss(
                    tar_smpl[hidx].betas, est_smpl[hidx].betas, device)
                ld['pseudogt_scale_losses'+h] += ((tar_smpl[hidx].scale - est_smpl[hidx].scale) ** 2) * \
                    self.pseudogt_shape_weight

            # pose prior losses for each human
            if self.pseudogt_transl_weight > 0:
                ld['pseudogt_transl_losses'+h] += self.get_pseudogt_transl_loss(
                    tar_smpl[hidx].transl, est_smpl[hidx].transl, device)

            # vertex to vertex losses for each human
            if self.pseudogt_v2v_weight > 0:
                ld['pseudogt_v2v_losses'+h] += self.get_pseudogt_v2v_loss(
                    tar_smpl[hidx].vertices, est_smpl[hidx].vertices, device)
            
            # joint to joint losses for each human
            if self.pseudogt_j2j_weight > 0:
                ld['pseudogt_j2j_losses'+h] += self.get_pseudogt_j2j_loss(
                    tar_smpl[hidx].joints, est_smpl[hidx].joints, device)

        # average the losses over batch
        ld_out = {}
        for k, v in ld.items():
            if type(v) == torch.Tensor:
                ld_out[k] = v.mean()

        # final loss value
        total_loss = sum(ld_out.values())
        # set breakpoint if total_loss is nan
        if torch.isnan(total_loss):
            import ipdb; ipdb.set_trace()
        ld_out['total_loss'] = total_loss

        return total_loss, ld_out
