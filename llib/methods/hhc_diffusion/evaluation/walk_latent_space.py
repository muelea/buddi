# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import argparse
import glob
import os
import os.path as osp
import imageio
import pickle
import torch
import numpy as np

from llib.defaults.main import config as default_config, merge as merge_configs
from llib.visualization.diffusion_eval import save_gif, render_images
from llib.visualization.scripts.tools import build_renderer, render_360_views

from llib.methods.hhc_diffusion.evaluation.utils import *
from llib.methods.hhc_diffusion.evaluation.eval import eval_diffusion

CUSTOM_SCHEDULE = {
    -1: [999, 750, 500, 250, 100, 60, 40, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    -2: [int(1000 * x * x) for x in np.arange(0.04, 1.0, 0.0096)[::-1]]
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-cfg",
        type=str,
        dest="exp_cfgs",
        nargs="+",
        default=None,
        help="The configuration of the experiment",
    )
    parser.add_argument(
        "--exp-opts",
        default=[],
        dest="exp_opts",
        nargs="*",
        help="The configuration of the Detector",
    )

    parser.add_argument("--eval-dataset-name", type=str, default="flickrci3ds")
    parser.add_argument("--eval-dataset-split", type=str, default="val")
    parser.add_argument(
        "--condition",
        action="store_true",
        help="pass in inpainting params as conditions",
    )
    parser.add_argument(
        "--inpaint", default=False, action="store_true", help="Inpaint second person."
    )
    parser.add_argument(
        "--inpaint-params",
        default=["orient_h0", "pose_h0", "shape_h0", "transl_h0"],
        nargs="*",
        help="Specified params will be fixed during sampling. Non-specified params will be generated.",
    )
    parser.add_argument(
        "--inpaint-item-idx",
        default=0,
        type=int,
        help="The index of the item in batch selected.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Folder where results will be saved. If None output folder is set to exp_cfg folder.",
    )
    parser.add_argument(
        "--vis-folder",
        default=None,
        help="Folder where results will be rendered. If None, will use output folder",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="latest",
        help="The model checkpoint to use for evaluation. If set to last, the latest checkpoint is used.",
    )
    parser.add_argument(
        "--num-samples",
        default=128,
        type=int,
        help="number of samples to generate (two batches)",
    )
    parser.add_argument(
        "--max-t",
        type=int,
        default=1000,
        help="The largest t to start the diffusion process from.",
    )
    parser.add_argument(
        "--log-steps",
        default=1,
        type=int,
        help="steps to log in result folder in visulize in gif",
    )
    parser.add_argument(
        "--skip-steps",
        default=1,
        type=int,
        help="skip skip n steps to next t every x steps. E.g. max-t = 1000 and skip-steps = 10 would set t to 1000, 1090, 1080, etc.",
    )
    parser.add_argument(
        "--body-model-utils-folder",
        type=str,
        default="essentials/body_model_utils/",
        help="SMPL folder o compute metrics",
    )
    parser.add_argument(
        "--max-images-render",
        default=0,
        type=int,
        help="number of images out of num-samples to render and save in output folder",
    )
    parser.add_argument(
        "--save-vis",
        default=False,
        action="store_true",
        help="save gif of the diffusion process.",
    )
    parser.add_argument(
        "--run-eval",
        default=False,
        action="store_true",
        help="Run the evaluation on a sampled batch.",
    )
    parser.add_argument("--batch_size", default=16, type=int)

    cmd_args = parser.parse_args()

    cfg = merge_configs(cmd_args, default_config)

    # for evaluation / comparisons, specify the dataset to use
    cfg = update_datasets(cfg, cmd_args.eval_dataset_split, cmd_args.eval_dataset_name)

    # set output folder to results if not specified
    if cmd_args.output_folder is None:
        cmd_args.output_folder = cmd_args.exp_cfgs[0].replace("config.yaml", "results")

    cfg = update_output_folder(cfg, cmd_cfg=cmd_args.exp_cfgs[0])

    # update batch size
    cfg.batch_size = cmd_args.batch_size

    cfg = update_paths(cfg)

    return cfg, cmd_args


def update_output_folder(cfg, cmd_cfg):
    exp_cfg_split = cmd_cfg.split("/")
    new_base_folder = "/".join(exp_cfg_split[:-2])
    cfg["logging"]["base_folder"] = new_base_folder
    cfg["logging"]["run"] = exp_cfg_split[-2]
    return cfg


def update_datasets(cfg, split, name):
    for ss in ["train", "val", "test"]:
        if ss == split:
            cfg["datasets"][f"{split}_names"] = [name]
            if ss == "train":
                cfg["datasets"][f"{split}_composition"] = [1.0]
        else:
            cfg["datasets"][f"{ss}_names"] = []
    return cfg


def save_result(OUTPUT_FOLDER, x_ts, x_starts):
    # write x_starts to file with pickle
    x_starts = move_to(x_starts, "cpu")
    x_ts = move_to(x_ts, "cpu")
    with open(f"{OUTPUT_FOLDER}/x_starts_smplx.pkl", "wb") as f:
        pickle.dump(x_starts, f)

    with open(f"{OUTPUT_FOLDER}/x_ts_smplx.pkl", "wb") as f:
        pickle.dump(x_ts, f)


def dump_cmd(OUTPUT_FOLDER, cmd_args, extra_args):
    # save cmd arguments and argv to file
    with open(f"{OUTPUT_FOLDER}/cmd_args.txt", "w") as f:
        f.write("cmd_args:" + str(cmd_args) + "\n")
        for k, v in extra_args.items():
            f.write(k + ": " + str(v))


def create_output_folder(
    OUTPUT_FOLDER, MAX_T, SKIP_STEPS, INPAINT, CONDITION, INPAINT_PARAMS, INPAINT_IDX,
):
    if INPAINT:
        sampling_mode = "inpaint"
        if INPAINT_IDX is not None:
            sampling_mode = f"{sampling_mode}_idx_{INPAINT_IDX}"
        if INPAINT_PARAMS is not None:
            sampling_mode = f"{sampling_mode}_fix_{'-'.join(INPAINT_PARAMS)}"
        else:
            sampling_mode = "{sampling_mode}_fix_all"
    else:
        sampling_mode = "generate"
    if CONDITION:
        sampling_mode = f"cond_{sampling_mode}"
    OUTPUT_FOLDER = osp.join(OUTPUT_FOLDER, f"{sampling_mode}_{MAX_T}_{SKIP_STEPS}")
    # count number of past samples we've done with these parameters
    num_matches = len(glob.glob(f"{OUTPUT_FOLDER}_v[0-9]*/"))
    print(f"FOUND {num_matches} matches for {OUTPUT_FOLDER}")
    OUTPUT_FOLDER = f"{OUTPUT_FOLDER}_v{num_matches}"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    return OUTPUT_FOLDER


def sample_from_ground_truth_dataset(num_batches, cfg, diffusion_module, item_idx=None):
    data, data_loader = setup_gt_dataset(cfg, drop_last=True)

    pgt_batches, pgt_verts = [], []
    for bidx, in_batch in enumerate(data_loader):
        if len(pgt_batches) >= num_batches:
            break
        batch = dict_to_device(in_batch, cfg.device)
        pgt_batch = diffusion_module.preprocess_batch(batch, in_data="pgt")
        pgt_batch = diffusion_module.cast_smpl(pgt_batch)
        if item_idx is not None:
            for param_name in ["orient", "pose", "shape", "transl"]:
                picked_val = pgt_batch[param_name][[item_idx]]
                pgt_batch[param_name][:] = picked_val

        pgt_smpl = diffusion_module.get_smpl(pgt_batch)
        verts = torch.stack(
            [pgt_smpl[0].vertices, pgt_smpl[1].vertices], dim=1
        )  # (B, 2, V, 3)

        pgt_verts.append(verts.cpu())
        pgt_batch = move_to(pgt_batch, "cpu")
        pgt_batches.append(pgt_batch)

    pgt_verts = torch.cat(pgt_verts, dim=0)
    print(len(pgt_batches), pgt_verts.shape)

    return pgt_batches, pgt_verts


def generate_inpainting_mask(
    params, config=["orient_h0", "pose_h0", "shape_h0", "transl_h0"]
):
    num_humans = params["orient"].shape[1]

    mask = {}
    for k in ["orient", "pose", "shape", "transl"]:
        mask[k] = torch.ones_like(params[k]).to(torch.bool)

    for kk in ["orient", "pose", "shape", "transl"]:
        for hh in range(2):
            if f"{kk}_h{hh}" in config:
                mask[kk][:, hh, :] = True
            else:
                mask[kk][:, hh, :] = False

    return mask


def save_batch_gifs(
    output_folder, renderer, verts, faces, suffix="", num_poses=30, fps=6, **kwargs
):
    """
    :param output_folder
    :param renderer pyrender renderer
    :param verts (B, 2, V, 3)
    :param faces (F, 3)
    :param num_poses (default 30)
    :param fps (default 6)
    """
    os.makedirs(output_folder, exist_ok=True)
    for i in range(len(verts)):
        frames = render_360_views(renderer, verts[i], faces, num_poses, **kwargs)
        imageio.mimwrite(f"{output_folder}/{i:05d}{suffix}.gif", frames, fps=fps)
    print(f"SAVED {len(verts)} RENDERS TO {output_folder}")

def save_batch_walk_gifs(
    output_folder, renderer, verts, faces, suffix="", num_poses=30, fps=6, **kwargs
):
    """
    :param output_folder
    :param renderer pyrender renderer
    :param verts (B, 2, V, 3)
    :param faces (F, 3)
    :param num_poses (default 30)
    :param fps (default 6)
    """
    os.makedirs(output_folder, exist_ok=True)
    for pp in range(len(verts)):
        for i in range(len(verts[pp])):
            frames = render_360_views(renderer, verts[pp][i], faces, num_poses, **kwargs)
            imageio.mimwrite(f"{output_folder}/{pp:05}_{i:05d}_{suffix}.gif", frames, fps=fps)
    print(f"SAVED RENDERS TO {output_folder}")



def walk_latent_space(cfg, cmd_args, diffusion_module):

    num_batches = 2 * round(cmd_args.num_samples / cmd_args.batch_size)

    T = (
        CUSTOM_SCHEDULE[cmd_args.max_t] if cmd_args.max_t < 0
        else np.arange(1, cmd_args.max_t, cmd_args.skip_steps)[::-1]
    )
    if cmd_args.max_t == -1:
        timestep_latent = 60
    elif cmd_args.max_t == -2:
        timestep_latent = 49
    else:
        timestep_latent = 51
    timestep_idx = np.where(np.array(T) == timestep_latent)[0][0]

    # sample two batches with start and end pose each
    start_end_frames = batch_sample(
        num_batches,
        diffusion_module,
        T,
        cmd_args.log_steps,
        conditions=None,
        condition_params=cmd_args.inpaint_params,
        sampling_function=sample_unconditional_latent,
        return_latent_vec=True
    )

    # select two samples 
    x_ts_all, x_starts_all, x_latents_all = start_end_frames
    num_samples = 20 #len(x_latents_all[51])
    all_verts = []
    for ii in range(num_samples):
        sample_00, sample_01 = x_latents_all[timestep_latent][ii], x_latents_all[timestep_latent][ii+1]
        interpolated_latent = []
        for k in np.linspace(0, 1, diffusion_module.body_model.batch_size):
            interpolated_latent.append(sample_00*(1-k) + sample_01*k)

        predictions = diffusion_module.model.unembed_input(
            diffusion_module.model.x_keys, torch.stack(interpolated_latent))

        denoised_params = diffusion_module.concat_humans(predictions)
        denoised_smpls = diffusion_module.get_smpl(denoised_params)
        timesteps = T[timestep_idx:] #np.arange(1, timestep_latent, 10)[::-1] #[41, 31, 21, 11, 1]
        x_ts, x_starts = diffusion_module.sampling_loop(
            x=denoised_smpls, y={}, ts=timesteps, inpaint=None, log_steps=1
        )
        x1, x2 = x_starts['final'][0].vertices.unsqueeze(1), x_starts['final'][1].vertices.unsqueeze(1)
        
        # predicted tokens to params and smpl bodies
        #denoised_params = diffusion_module.concat_humans(predictions)
        #denoised_smpls = diffusion_module.get_smpl(denoised_params)

        # concatenate along first dimension
        #x1, x2 = denoised_smpls[0].vertices.unsqueeze(1), denoised_smpls[1].vertices.unsqueeze(1)
        # concatenate x1, x2 along first dimension
        vertices = torch.cat([x1, x2], dim=1)
        all_verts.append(vertices)
    
    return all_verts

@torch.no_grad()
def main(cfg, cmd_args):
    MAX_T = cmd_args.max_t
    OUTPUT_FOLDER = cmd_args.output_folder
    SKIP_STEPS = cmd_args.skip_steps
    MAX_IMAGES = cmd_args.max_images_render
    INPAINT = cmd_args.inpaint

    T = CUSTOM_SCHEDULE[MAX_T] if MAX_T < 0 else np.arange(1, MAX_T, SKIP_STEPS)[::-1]
    OUTPUT_FOLDER = create_output_folder(
        OUTPUT_FOLDER,
        MAX_T,
        SKIP_STEPS,
        INPAINT,
        cmd_args.condition,
        cmd_args.inpaint_params,
        cmd_args.inpaint_item_idx,
    )
    dump_cmd(OUTPUT_FOLDER, cmd_args, extra_args={"T": T})


    VIS_FOLDER = f"{OUTPUT_FOLDER}/renders"
    os.makedirs(VIS_FOLDER, exist_ok=True)

    diffusion_module = setup_diffusion_module(cfg, cmd_args)

    vertices = walk_latent_space(cfg, cmd_args, diffusion_module)

    renderer = build_renderer()
    faces = diffusion_module.faces_tensor.to("cpu")
    save_batch_walk_gifs(VIS_FOLDER, renderer, vertices, faces, suffix="_walk")



if __name__ == "__main__":
    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
