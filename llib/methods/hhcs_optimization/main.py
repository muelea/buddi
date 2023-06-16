import argparse
import torch 
import numpy as np
import cv2
import os
import sys
import smplx
import pickle
import random
import trimesh
import imageio
from tqdm import tqdm
import os.path as osp
from loguru import logger as guru
from omegaconf import OmegaConf 

from llib.bodymodels.build import build_bodymodel 
from llib.cameras.build import build_camera
from llib.data.build import build_optimization_datasets
from llib.visualization.renderer import Pytorch3dRenderer
from llib.visualization.utils import *
from llib.logging.logger import Logger
from llib.defaults.main import (
    config as default_config,
    merge as merge_configs
)
from llib.models.build import build_model
from llib.models.diffusion.build import build_diffusion

from loss_module import HHCOptiLoss
from fit_module import HHCSOpti

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True 

DEBUG_IMG_NAMES = []

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg', 
        type=str, dest='exp_cfgs', nargs='+', default=None, 
        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
        nargs='*', help='The configuration of the Detector') 
    parser.add_argument('--cluster_pid', 
        type=str, default=None, help='Cluster process id')
    parser.add_argument('--cluster_bs', 
        type=int, default=None, help='cluster batch size')

    cmd_args = parser.parse_args()

    cfg = merge_configs(cmd_args, default_config)

    return cfg, cmd_args

def main(cfg, cmd_args):

    is_cluster = cmd_args.cluster_pid is not None

    # cluster configuration
    if is_cluster:
        cpid = int(cmd_args.cluster_pid)
        cbs = int(cmd_args.cluster_bs)
        c_item_idxs = np.arange(cpid*cbs, cpid*cbs+cbs)

    # save config file and create output folders
    logger = Logger(cfg)

    # make sure only one dataset is used
    assert len(cfg.datasets.train_names) <= 1, \
        "Only one dataset is supported for optimization. Hint: change config.datasets.train_names."

    # create dataloader
    FITTING_DATASETS = build_optimization_datasets(
        datasets_cfg=cfg.datasets,
        body_model_type=cfg.body_model.type, # necessary to load the correct contact maps
    )
    
    for ds in FITTING_DATASETS:
        if ds is None:
            continue

        guru.info(f'Processing {len(ds)} items from {ds.dataset_name}.')
        
        for item_idx in tqdm(range(len(ds))):
            if is_cluster:
                if item_idx not in c_item_idxs:
                    continue
            #try:
            if True:
                #guru.info(f'Processing item number {item_idx}')
                item = ds.get_single_item(item_idx)
                # check if item was already processed, if so, skip
                img_fn_out = item['imgname_fn_out']
                # keep to debug specific images
                """
                if img_fn_out not in DEBUG_IMG_NAMES:
                    continue
                """

                out_fn_res = osp.join(logger.res_folder, f'{img_fn_out}.pkl')
                out_fn_img = osp.join(logger.img_folder, f'{img_fn_out}.png')
                if osp.exists(out_fn_res) and osp.exists(out_fn_img):
                    guru.info(f'Item {img_fn_out} was already processed. Skipping.')
                else:
                    process_item(cfg, item, logger)
            #except Exception as e:
            #    guru.exception(e)
            #    # if exception is keyboard interrupt, stop
            #    if isinstance(e, KeyboardInterrupt):
            #        sys.exit()

def process_item(cfg, item, logger):

    img_fn_out = item['imgname_fn_out']
    out_fn_res = osp.join(logger.res_folder, f'{img_fn_out}.pkl')
    out_fn_img = osp.join(logger.img_folder, f'{img_fn_out}.png')
    out_fn_gif = osp.join(logger.sum_folder, f'{img_fn_out}.gif')

    # configuration for optimization
    opti_cfg = cfg.model.optimization

    # load body models for human1 and human2
    body_model_h1 = build_bodymodel(
        cfg=cfg.body_model, 
        batch_size=cfg.batch_size, 
        device=cfg.device
    )
    body_model_h2 = build_bodymodel(
        cfg=cfg.body_model, 
        batch_size=cfg.batch_size, 
        device=cfg.device
    )

    # build regressor used to predict diffusion params
    if opti_cfg.use_diffusion:
        from llib.methods.hhc_diffusion.train_module import TrainModule
        
        diffusion_cfg = default_config.copy()
        diffusion_cfg.merge_with(OmegaConf.load(opti_cfg.pretrained_diffusion_model_cfg))
        #diffusion_logger = Logger(diffusion_cfg)
        regressor = build_model(diffusion_cfg.model.regressor).to(cfg.device)
        #checkpoint = torch.load(diffusion_logger.get_latest_checkpoint())
        checkpoint = torch.load(opti_cfg.pretrained_diffusion_model_ckpt)
        regressor.load_state_dict(checkpoint['model'], strict=False)
        diffusion = build_diffusion(**diffusion_cfg.model.diffusion)
        body_model = build_bodymodel(
            cfg=cfg.body_model, 
            batch_size=diffusion_cfg.batch_size, 
            device=cfg.device
        )
        diffusion_module = TrainModule(
            cfg=diffusion_cfg,
            train_dataset=None,
            val_dataset=None,
            diffusion=diffusion,
            model=regressor,
            criterion=None,
            evaluator=None,
            body_model=body_model,
            renderer=None,
        ).to(cfg.device)
    else:
        diffusion_module = None

    # create camera
    camera = build_camera(
        camera_cfg=cfg.camera,
        camera_type=cfg.camera.type,
        batch_size=cfg.batch_size,
        device=cfg.device
    ).to(cfg.device)

    # create losses
    criterion = HHCOptiLoss(
        losses_cfgs = opti_cfg.losses,
        body_model_type = cfg.body_model.type,
    ).to(cfg.device)

    # create optimizer module
    optimization_module = HHCSOpti(
        opti_cfg=opti_cfg,
        camera=camera,
        body_model_h1=body_model_h1,
        body_model_h2=body_model_h2,
        criterion=criterion,
        batch_size=cfg.batch_size,
        device=cfg.device,
        diffusion_module=diffusion_module,
    )

    # transform input item to human1, human2 and camera dict
    h0_data, h1_data, cam_data = {}, {}, {}
    for k, v in item.items():
        if '_h0' in k:
            new_key = k.replace('_h0', '')
            h0_data[new_key] = v.to(cfg.device)
        elif '_h1' in k:
            new_key = k.replace('_h1', '')
            h1_data[new_key] = v.to(cfg.device)
        elif k in ['pitch', 'yaw', 'roll', 'tx', 'ty', 'tz', 'fl', 'ih', 'iw']:
            cam_data[k] = v.to(cfg.device)
        else:
            pass

    # set contact map to none if diffusion is being used a prior 
    contact_map = item['contact_map'] if opti_cfg.use_gt_contact_map \
        else torch.zeros_like(item['contact_map'])
        
    # the last shape component is used to interpolate between SMIL and SMPL-X
    # since the value is not normalized, we enfore it to be between 0 and 1
    # using a sigmoid function

    # optimize each item in dataset
    smpl_output_h1, smpl_output_h2 = optimization_module.fit(
        init_h1=h0_data,
        init_h2=h1_data,
        init_camera=cam_data,
        contact_map=contact_map,
    )

    guru.info(f'Optimization finished for {img_fn_out}. Saving results.')

    # save meshes 
    v1 = smpl_output_h1.vertices.detach().cpu().numpy()[0]
    v2 = smpl_output_h2.vertices.detach().cpu().numpy()[0]
    m1 = trimesh.Trimesh(v1, body_model_h1.faces)
    m2 = trimesh.Trimesh(v2, body_model_h2.faces)
    _ = m1.export(osp.join(logger.res_folder, f'{img_fn_out}_h1.obj'))
    _ = m2.export(osp.join(logger.res_folder, f'{img_fn_out}_h2.obj'))
    
    # visualize results
    renderer = Pytorch3dRenderer(
        cameras = camera.cameras,
        image_width=item['img_width'],
        image_height=item['img_height'],
    )
    renderer_newview = Pytorch3dRenderer(
        cameras = camera.cameras,
        image_width=200,
        image_height=300,
    )

    verts_fit = torch.cat([smpl_output_h1.vertices, smpl_output_h2.vertices], dim=0)
    verts_bev = torch.cat([item['bev_orig_vertices_h0'].unsqueeze(0), 
        item['bev_orig_vertices_h1'].unsqueeze(0)], dim=0).to(cfg.device)
    smplx_faces = torch.from_numpy(body_model_h1.faces.astype(np.int32)).to(cfg.device)
    # create smpl model to get smpl faces
    smpl_body_model = smplx.create(
        model_path=cfg.body_model.smpl_family_folder,
        model_type='smpl'
    )
    smpl_faces = torch.from_numpy(smpl_body_model.faces.astype(np.int32)).to(cfg.device)

    vertices_methods = [verts_fit, verts_bev]
    colors = [['light_blue3', 'light_blue5'], ['light_red3', 'light_red5']]
    imgpath = item['imgpath']
    orig_img = cv2.imread(imgpath)[:,:,::-1].copy().astype(np.float32)
    IMG = add_alpha_channel(orig_img)

    # add keypoints to image
    IMGORIG = IMG.copy()
    IMGBEV = IMG.copy()
    h1pp = camera.project(smpl_output_h1.joints)
    h2pp = camera.project(smpl_output_h2.joints)
    for idx, joint in enumerate(h1pp[0]):
        IMGBEV = cv2.circle(IMGBEV, (int(joint[0]), int(joint[1])), 3, (255, 255, 0), 2)
    for idx, joint in enumerate(h2pp[0]):
        IMGBEV = cv2.circle(IMGBEV, (int(joint[0]), int(joint[1])), 3, (255, 255, 0), 2)

    # add keypoints to image
    h1_kp2d = item['vitpose_keypoints_h0']
    h2_kp2d = item['vitpose_keypoints_h1']
    keypoints_list = [item['vitpose_keypoints_h0'], item['vitpose_keypoints_h1'], 
    item['op_keypoints_h0'], item['op_keypoints_h1']]
    col = [(255, 0, 0), (0, 255, 0), (125, 0, 0), (0, 125, 0)]
    for idx, kp2d in enumerate(keypoints_list):
        for joint in kp2d:
            IMGBEV = cv2.circle(IMGBEV, (int(joint[0]), int(joint[1])), 3, col[idx], 2)

    h1_bbox = item['bbox_h0'].numpy().astype(np.int32)
    h2_bbox = item['bbox_h1'].numpy().astype(np.int32)
    IMGBEV = cv2.rectangle(IMGBEV, (h1_bbox[0], h1_bbox[1]), (h1_bbox[2], h1_bbox[3]), (255, 0, 0), 2)
    IMGBEV = cv2.rectangle(IMGBEV, (h2_bbox[0], h2_bbox[1]), (h2_bbox[2], h2_bbox[3]), (0, 255, 0), 2)

    imgs_out = []
    for vidx, (verts, meshcol) in enumerate(zip(vertices_methods, colors)):
        faces = smpl_faces if verts.shape[1] == 6890 else smplx_faces
        bm = 'smpl' if verts.shape[1] == 6890 else 'smplx'
        IMG = IMGBEV.copy() if verts.shape[1] == 6890 else IMGORIG.copy()
        renderer.update_camera_pose(
            camera.pitch.item(), camera.yaw.item(), camera.roll.item(), 
            camera.tx.item(), camera.ty.item(), camera.tz.item()
        )
        rendered_img = renderer.render(verts, faces, colors = meshcol, body_model=bm)
        color_image = rendered_img[0].detach().cpu().numpy() * 255
        overlay_image = overlay_images(IMGORIG.copy(), color_image)
        image_out = np.hstack((IMG, overlay_image))

        # now with different views
        vertex_transl_center = verts.mean((0,1))
        #vertex_transl_center[1] = 0 # translate only on the ground plane
        verts_centered = verts - vertex_transl_center
        # y-axis rotation
        for yy in [45.0, 90.0, 135.0]:
            renderer_newview.update_camera_pose(0.0, yy, 180.0, 0.0, 0.2, 2.0)
            rendered_img = renderer_newview.render(
                verts_centered, faces, colors = meshcol, body_model=bm)
            color_image = rendered_img[0].detach().cpu().numpy() * 255
            scale = image_out.shape[0] / color_image.shape[0]
            newsize = (int(scale * color_image.shape[1]), int(image_out.shape[0]))
            color_image = cv2.resize(color_image, dsize=newsize)
            image_out = np.hstack((image_out, color_image))
        
        # bird view
        for pp in [270.0]:
            renderer_newview.update_camera_pose(pp, 0.0, 180.0, 0.0, 0.0, 2.0)
            rendered_img = renderer_newview.render(verts_centered, faces, colors = meshcol, body_model=bm)
            color_image = rendered_img[0].detach().cpu().numpy() * 255
            scale = image_out.shape[0] / color_image.shape[0]
            newsize = (int(scale * color_image.shape[1]), int(image_out.shape[0]))
            color_image = cv2.resize(color_image, dsize=newsize)
            image_out = np.hstack((image_out, color_image))
        imgs_out.append(image_out)

    image_out = np.vstack((imgs_out[0], imgs_out[1]))
    cv2.imwrite(out_fn_img, image_out[...,[2,1,0,3]][...,:3])

    # save results 
    output = {
        'h0': {},
        'h1': {},
        'cam': {},
    }
    for k, v in body_model_h1.named_parameters():
        if v is not None:
            output['h0'][k] = v.detach().cpu().numpy()
        else:
            print(k,v)
    for k, v in body_model_h2.named_parameters():
        if v is not None:
            output['h1'][k] = v.detach().cpu().numpy()
        else:
            print(k,v)
    for k, v in camera.named_parameters():
        if v is not None:
            output['cam'][k] = v.detach().cpu().numpy()
        else:
            print(k,v)
            
    with open(out_fn_res, 'wb') as f:
        pickle.dump(output, f)


    # render noise / denoise
    if len(optimization_module.criterion.debug) > 0: 
        color_images = []
        renderer_newview.update_camera_pose(0.0, 0.0, 180.0, 0.0, 0.2, 2.0)
        for verts_all in optimization_module.criterion.debug:
            for verts in verts_all:
                vertex_transl_center = verts.mean((0,1))
                verts_centered = verts - vertex_transl_center
                rendered_img = renderer_newview.render(
                    verts_centered, smplx_faces, 
                    colors = ['light_red3', 'light_red5'], 
                    body_model='smplx')
                color_image = rendered_img[0].detach().cpu().numpy() * 255
                scale = image_out.shape[0] / color_image.shape[0]
                newsize = (int(scale * color_image.shape[1]), int(image_out.shape[0]))
                color_image = cv2.resize(color_image, dsize=newsize)
                color_images.append(color_image.astype(np.uint8))
        
        # save gif of noise / denoise
        imageio.mimsave(out_fn_gif, color_images, fps=5)

if __name__ == "__main__":
    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
