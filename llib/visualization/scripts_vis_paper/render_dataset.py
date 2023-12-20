import argparse
import os
import tqdm
import pickle
import numpy as np
import torch
import imageio
import glob
import shutil
import smplx
from omegaconf import OmegaConf

from llib.defaults.main import config as default_config, merge as merge_configs
from llib.bodymodels.build import build_bodymodel
from llib.visualization.scripts.tools import (
    move_to,
    render_camera_view,
    render_top_view,
    render_360_views,
    build_renderer,
    crop_image_from_alpha,
)


def get_data_field(item, prefix, field):
    """
    returns h0 and h1 elements (*) stacked into (2, *)
    """
    #if field == "scale" and f"{prefix}_{field}_h0" not in item:
    #    return torch.zeros(2, 1)

#    return torch.stack(
#        [torch.as_tensor(item[f"{prefix}_{field}_h{i}"]) for i in range(2)], dim=0
#    )

    return torch.as_tensor(item[f"{prefix}_{field}"])

# def get_stacked_element_fit(item, field):
#     """
#     returns h0 and h1 elements (*) stacked into (2, *)
#     """
#     human_ids = ["h0", "h1"]
#     return torch.cat([torch.as_tensor(item[hid][field]) for hid in human_ids], dim=0)


def get_pseudo_gt_vertices(item, body_model, device="cpu", prefix="pseudogt"):
    """
    Read SMPL parameters and run SMPL forward pass to get vertices.
    returns (2, V, 3)
    """
    fields = [
        "betas",
        "scale",
        "global_orient",
        "body_pose",
        "transl",
    ]
    target = {f: move_to(get_data_field(item, prefix, f), device) for f in fields}
    # target["scale"] = -1.0 / (1.0 + torch.exp((target["scale"] - 0.5) * 20)) + 1.0
    with torch.no_grad():
        body = body_model(**target)
    return body.vertices.detach().cpu()


def get_pseudo_gt_vertices_all(path, body_model, device, selection=None):
    if not os.path.isfile(path):
        raise ValueError(f"{path} does not exist!!")

    with open(path, "rb") as f:
        data = pickle.load(f)

    # data is a list
    out_dict = {}
    for res in tqdm.tqdm(data):
        imgname = res["imgname"]
        name = os.path.splitext(imgname)[0]
        contact_index = res["contact_index"]
        name = f"{name}_{contact_index}"
        if selection is not None and name not in selection:
            continue
        verts = get_pseudo_gt_vertices(res, body_model, device, prefix='pgt_smplx')
        iw, ih, fl = res["img_width"], res["img_height"], res["fl"]
        intrins = (fl, fl, iw / 2, ih / 2, iw, ih)
        out_dict[name] = {
            "vertices": {name: verts},
            "intrins": intrins,
            "imgname": imgname,
        }
    return out_dict


def get_bev_vertices(item, body_model, device, prefix="bev_orig"):
    """
    Read SMPL parameters and run SMPL forward pass to get vertices.
    returns (2, V, 3)
    """
    vertices = get_data_field(item, prefix, "vertices")

    # swap h0 and h1 based on translaiton in x direction
    if item['transl_h0'][0] > item['transl_h1'][0]:
        vertices = vertices.flip(0)

    return vertices


def get_bev_vertices_all(path, body_model, device, selection=None):
    if not os.path.isfile(path):
        raise ValueError(f"{path} does not exist!!")

    with open(path, "rb") as f:
        data = pickle.load(f)

    # data is a list
    out_dict = {}
    for res in tqdm.tqdm(data):
        imgname = res["imgname"]
        name = os.path.splitext(imgname)[0]
        contact_index = res["contact_index"]
        name = f"{name}_{contact_index}"
        if selection is not None and name not in selection:
            continue
        verts = torch.from_numpy(res['bev_smpl_vertices_root_trans'])
        #verts = get_bev_vertices(res, body_model, device)
        iw, ih, fl = res["img_width"], res["img_height"], res["fl"]
        intrins = (fl, fl, iw / 2, ih / 2, iw, ih)
        out_dict[name] = {
            "vertices": {name: verts},
            "intrins": intrins,
            "imgname": imgname,
        }

    return out_dict


def get_fit_vertices(item, body_model, device):
    """
    Read SMPL parameters and run SMPL forward pass to get vertices.
    returns (2, V, 3)
    """
    fields = [
        "betas",
        "scale",
        "global_orient",
        "body_pose",
        "transl",
    ]
    target = {f: move_to(item[f], device) for f in fields}
    body = body_model(**target)
    return body.vertices.detach().cpu()


def get_fit_vertices_all(res_dir, img_dir, body_model, device, selection=None):
    files = glob.glob(os.path.join(res_dir, "*.pkl"))

    # data is a list
    out_dict = {}
    #verts_dict = {}
    for file in tqdm.tqdm(files):
        name = os.path.splitext(os.path.basename(file))[0]
        src_img_name = '_'.join(name.split('_')[:-1]) + '.png'
        src_img = imageio.imread(os.path.join(img_dir, src_img_name))
        ih, iw, _ = src_img.shape
        if selection is not None and name not in selection:
            continue
        data = pickle.load(open(file, "rb"))
        verts = get_fit_vertices(data['humans'], body_model, device)
        verts_dict = {name: verts}
        # cannot render source view with output file info, omitting intrins and imgname
        fl = data['cam']["fl"].item()
        intrins = (fl, fl, iw / 2, ih / 2, iw, ih)
        out_dict[name] = {
            "vertices": verts_dict,
            "intrins": intrins,
            #"extrinsics": data["cam"],
            "imgname": src_img_name,
            #"srcimage": src_img,
        }
    return out_dict


def render_data_dict(
    render_dict,
    renderer,
    body_model,
    img_dir,
    out_dir,
    selection=None,
    save_src=False,
    save_top=False,
    save_gif=False,
    save_side=False,
    num_poses=30,
    fps=6,
    **kwargs,
):
    if not (save_src or save_top or save_gif):
        print("no render views selected, skipping")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    faces = torch.as_tensor(body_model.faces.astype(np.int32)).cpu()
    orig_intrins = renderer.get_intrinsics()
    for out_name, item_dict in render_dict.items():
        imgname = item_dict["imgname"]
        cur_intrins = item_dict["intrins"]
        if selection is not None and out_name not in selection:
            continue

        if save_src:
            src_img = imageio.imread(os.path.join(img_dir, imgname))

        print(out_name)

        for src_name, verts in item_dict["vertices"].items():
            out_sub = f"{out_dir}/{src_name}"
            os.makedirs(out_sub, exist_ok=True)

            if save_src:
                renderer.update_intrinsics(*cur_intrins)
                frame = render_camera_view(renderer, verts, faces, src_img)
                imageio.imwrite(f"{out_sub}/{out_name}_src.png", frame)

                # render vertices only to get RGBA
                alpha = render_camera_view(renderer, verts, faces)[...,[-1]]
                frame_crop, _ = crop_image_from_alpha(frame, alpha, to_square=True, pad_factor=0.1)
                imageio.imwrite(f"{out_sub}/{out_name}_src_crop.png", frame_crop)
                src_img_crop, _ = crop_image_from_alpha(src_img, alpha, to_square=True, pad_factor=0.1)
                imageio.imwrite(f"{out_sub}/{out_name}_src_img_crop.png", src_img_crop)
                
            if save_top:
                renderer.update_intrinsics(*orig_intrins)
                frame = render_top_view(renderer, verts, faces)
                imageio.imwrite(f"{out_sub}/{out_name}_top.png", frame)

            if save_gif:
                renderer.update_intrinsics(*orig_intrins)
                frames = render_360_views(renderer, verts, faces, num_poses, **kwargs)
                imageio.mimwrite(f"{out_sub}/{out_name}.gif", frames, fps=fps)

            if save_side:
                renderer.update_intrinsics(*orig_intrins)
                frames = render_360_views(renderer, verts, faces, 7, **kwargs)
                imageio.imwrite(f"{out_sub}/{out_name}_side_00.png", frames[0])
                imageio.imwrite(f"{out_sub}/{out_name}_side_01.png", frames[1])
                imageio.imwrite(f"{out_sub}/{out_name}_side_05.png", frames[5])

BASE_DIR = os.path.abspath(f"{__file__}/..")
HHC_HOME = os.environ["HUMANHUMANCONTACT_HOME"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_root", default=f"{HHC_HOME}/datasets")
    parser.add_argument("-j", "--result_root", default=f"{HHC_HOME}/results")
    parser.add_argument("-img", "--image_root", default=f"{HHC_HOME}/datasets")
    parser.add_argument(
        "-o", "--out_root", default="/shared/vye/humanhumancontact/pgt_renders"
    )
    parser.add_argument("--data_type", default="flickr", choices=["flickr", "chi3d"])
    parser.add_argument("--data_split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--body_cfg", default=f"{BASE_DIR}/cfg_body_model.yaml")
    parser.add_argument("--render_cfg", default=f"{BASE_DIR}/cfg_render.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--src", action="store_true", help="save source view")
    parser.add_argument("--top", action="store_true", help="save top view")
    parser.add_argument("--side", action="store_true", help="save side views")
    parser.add_argument("--gif", action="store_true", help="save gif")
    parser.add_argument("--copy_src_images", action="store_true", 
        help='copy source image to output directory')
    parser.add_argument("--colors", default=None, nargs="*", 
        help="color for each view")
    parser.add_argument(
        "--method",
        default="pseudo_ground_truth",
        #choices=[
        #    "pseudo_ground_truth",
        #    "bev",
        #    "fit_baseline",
        #    "fit_pseudogt",
        #    "fit_diffprior",
        #],
    )
    parser.add_argument(
        "--selection",
        default=None,
        nargs="*",
        help="select specific images in dataset by name, e.g. girls_113749_0",
    )
    parser.add_argument(
        "--save_render_dict_only", default=False, action="store_true",
        help="Does not run the rendering code, but saves the render dict e.g. for further use with blender."
    )
    args = parser.parse_args()

    cfg = default_config.copy()
    cfg.merge_with(OmegaConf.load(args.body_cfg))
    render_cfg = OmegaConf.load(args.render_cfg)

    return cfg, render_cfg, args


def main(cfg, render_cfg, args):
    if not (args.src or args.top or args.gif):
        print("NO RENDERING SPECIFIED")
        return

    renderer = build_renderer(render_cfg.renderer)
    
    body_model = build_bodymodel(cfg=cfg.body_model, batch_size=2, device=args.device)

    if args.data_type == "flickr":
        data_name = "FlickrCI3D_Signatures"
    elif args.data_type == "chi3d":
        data_name = "CHI3D"
        if args.src:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # read image names from files if provided
    selection = args.selection
    if selection is not None and len(selection) == 1 and selection[0].endswith(".txt"):
        with open(selection[0], "r") as f:
            selection = [line.strip() for line in f.readlines()]

    img_split = "train" if args.data_split in ["train", "val"] else "test"
    img_dir = f"{args.data_root}/original/{data_name}/{img_split}/images"

    #data_path = f"{args.data_root}/processed/{data_name}/{args.data_split}.pkl"
    if args.method == "bev":
        data_path = f"{args.data_root}/processed/{data_name}/{img_split}_optimization.pkl"
    elif args.method == "pseudo_ground_truth":
        data_path = f"{args.data_root}/processed/{data_name}/{args.data_split}_diffusion.pkl"
    
    out_dir = f"{args.out_root}/{data_name}/{args.data_split}/{args.method}"

    if args.method == "pseudo_ground_truth":
        render_dict = get_pseudo_gt_vertices_all(
            data_path, body_model, args.device, selection=selection
        )
    elif args.method == "bev":
        render_dict = get_bev_vertices_all(
            data_path, body_model, args.device, selection=selection
        )
    else: # args.method in ["fit_baseline", "fit_pseudogt", "fit_diffprior"]:
        #assert not args.src, "cannot save source views"
        img_dir = f"{args.image_root}/original/{data_name}/{img_split}/images"
        if args.copy_src_images:
            img_out_dir = f"{args.out_root}/{data_name}/{args.data_split}/src_images"
            os.makedirs(img_out_dir, exist_ok=True)
            for img_fn in selection:
                img_src_fn = '_'.join(img_fn.split('_')[:-1]) + '.png'
                img_path = os.path.join(img_dir, img_src_fn)
                shutil.copy(img_path, f"{img_out_dir}/{img_src_fn}")

        data_name_input = (
            "flickrci3ds" if args.data_type == "flickr" else args.data_type
        )
        data_folder = f"{args.result_root}/{args.method}_{data_name_input}_{args.data_split}/results"
        out_dir = f"{args.out_root}/{data_name}/{args.data_split}/{args.method}"
        render_dict = get_fit_vertices_all(
            data_folder, img_dir, body_model, args.device, selection=selection
        )
    #else:
    #    raise NotImplementedError

    # save data dict 
    # convert to blender compatibel format (remove tensors)
    if args.save_render_dict_only:
        data_out = {}
        for k, v in render_dict.items():
            data_out[k] = v 
            data_out[k]['vertices'][k] = render_dict[k]['vertices'][k].numpy()
        
        print('Save render_dict to ', out_dir)
        with open(f'{out_dir}/render_dict.pkl', 'wb') as f:
            pickle.dump(data_out, f)
    else:
        render_data_dict(
            render_dict,
            renderer,
            body_model,
            img_dir,
            out_dir,
            save_src=args.src,
            save_top=args.top,
            save_gif=args.gif,
            save_side=args.side,
            **render_cfg.vid_args,
        )


if __name__ == "__main__":
    cfg, render_cfg, args = parse_args()
    main(cfg, render_cfg, args)
