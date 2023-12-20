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

import matplotlib.pyplot as plt


CUSTOM_SCHEDULE = {
    -1: [999, 750, 500, 250, 100, 60, 40, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    -2: [int(1000 * x * x) for x in np.arange(0.04, 1.0, 0.0096)[::-1]]
}

def plot_schedule(numbers, out_name='schedule.png', size=(20,4)):
    fig, ax = plt.subplots(figsize=size)
    colors = ['r', 'g', 'b', 'y', 'k', 'm', 'c']
    for idx, num in enumerate(numbers):
        for x in num:
            ax.plot([x, x], [idx, idx+1], color=colors[idx%len(colors)])
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, len(numbers)+1])
    ax.yaxis.set_visible(False)
    # label / name x axis
    ax.set_xlabel('variance level')
    # set tick x axis every 1
    ax.set_xticks(np.arange(0, 1000, 20))
    # set x axis label every 100
    ax.set_xticklabels(np.arange(0, 1000, 20))
    plt.savefig(out_name)

# Example usage
#plot_schedule(
#    [CUSTOM_SCHEDULE[-1], CUSTOM_SCHEDULE[-2],np.arange(1, 1000, 10)[::-1]], 
#    out_name='schedules.png', size=(60,4)
#)


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
        "--eta",
        default=0.0,
        type=float,
        help="eta value to use for sampling",
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
    parser.add_argument(
        "--render-width",
        default=200,
        type=int,
        help="width of the rendered image",
    )
    parser.add_argument(
        "--render-height",
        default=256,
        type=int,
        help="height of the rendered image",
    )
    parser.add_argument(
        '--render-floor',
        default=False,
        action='store_true',
        help='render floor in visualization'
    )
    parser.add_argument(
        '--render-high-res',
        action='store_true',
        help='render high resolution images with floor'
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


def run_sampling(cfg, cmd_args, diffusion_module, eta=0.0):
    num_batches = round(cmd_args.num_samples / cmd_args.batch_size)

    if cmd_args.inpaint:
        param_batches, given_vertices = sample_from_ground_truth_dataset(
            num_batches, cfg, diffusion_module, item_idx=cmd_args.inpaint_item_idx
        )
        C = []
        for params in param_batches:
            mask = generate_inpainting_mask(params, cmd_args.inpaint_params)
            C.append(
                {"mask": mask, "values": params}
            )  # conditions / what will not be generated
        print("NUMBER OF CONDITION BATCHES", len(C))
        sampling_function = (
            sample_conditional_with_inpainting
            if cmd_args.condition
            else sample_unconditional_with_inpainting
        )
    else:
        sampling_function = sample_unconditional
        C = None  # conditions

    T = (
        CUSTOM_SCHEDULE[cmd_args.max_t]
        if cmd_args.max_t < 0
        else np.arange(1, cmd_args.max_t, cmd_args.skip_steps)[::-1]
    )
    output = batch_sample(
        num_batches,
        diffusion_module,
        T,
        cmd_args.log_steps,
        conditions=C,
        condition_params=cmd_args.inpaint_params,
        sampling_function=sampling_function,
        eta=eta
    )

    if cmd_args.inpaint:
        x_ts, x_starts = output
        return x_ts, x_starts, given_vertices
    else:
        return output


@torch.no_grad()
def main(cfg, cmd_args):
    MAX_T = cmd_args.max_t
    OUTPUT_FOLDER = cmd_args.output_folder
    LOG_STEPS = cmd_args.log_steps
    SKIP_STEPS = cmd_args.skip_steps
    MAX_IMAGES = cmd_args.max_images_render
    INPAINT = cmd_args.inpaint

    RENDER_HIGH_RES = cmd_args.render_high_res
    if RENDER_HIGH_RES:
        cmd_args.render_width = 800
        cmd_args.render_height = 1024
        cmd_args.render_floor = True

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

    VIS_FOLDER = cmd_args.vis_folder
    if cmd_args.save_vis:
        if VIS_FOLDER is not None:
            VIS_FOLDER = create_output_folder(
                VIS_FOLDER,
                MAX_T,
                SKIP_STEPS,
                INPAINT,
                cmd_args.condition,
                cmd_args.inpaint_params,
                cmd_args.inpaint_item_idx,
            )
        else:
            VIS_FOLDER = f"{OUTPUT_FOLDER}/renders"
            os.makedirs(VIS_FOLDER, exist_ok=True)

    diffusion_module = setup_diffusion_module(cfg, cmd_args)

    output = run_sampling(cfg, cmd_args, diffusion_module, eta=cmd_args.eta)

    if INPAINT:
        x_ts, x_starts, given_vertices = output
    else:
        x_ts, x_starts = output

    final_vertices = x_starts["final"]["vertices"]
    print("final verts shape", final_vertices.shape)
    save_result(OUTPUT_FOLDER, x_ts, x_starts)

    if cmd_args.save_vis:
        # render results
        renderer = build_renderer(width=cmd_args.render_width, height=cmd_args.render_height)
        faces = diffusion_module.faces_tensor.to("cpu")
        save_batch_gifs(VIS_FOLDER, renderer, final_vertices, faces, suffix="_gen", show_ground=cmd_args.render_floor)
    #     render_images(final_vertices, diffusion_module, MAX_IMAGES, OUTPUT_FOLDER)

    if INPAINT:
        # write vertices to file
        #print("given verts shape", given_vertices.shape)
        #inpaint_info = {"gt_vertices": given_vertices, "params": params, "mask": mask}
        #with open(f"{OUTPUT_FOLDER}/inpaint_info.pkl", "wb") as f:
        #    pickle.dump(inpaint_info, f)
        if cmd_args.save_vis:
            save_batch_gifs(VIS_FOLDER, renderer, given_vertices, faces, suffix="_gt", show_ground=cmd_args.render_floor)
    #         render_images(
    #             given_vertices, diffusion_module, 1, OUTPUT_FOLDER, img_prefix="gt_"
    #         )
    return
    # Render meshes and create gif ot each timestep
    if cmd_args.save_vis:
        x_ts.pop("final")
        x_starts.pop("final")
        save_gif(
            diffusion_module.renderer,
            diffusion_module,
            x_ts,
            MAX_T,
            f"{VIS_FOLDER}/meshes.gif",
            x_starts,
            max_images=4,
            is_batch=True,
        )


if __name__ == "__main__":
    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
