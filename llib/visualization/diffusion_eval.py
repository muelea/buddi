import matplotlib.pyplot as plt 
import shutil 
import os 
import numpy as np 
import cv2 
import torch 
import imageio
import itertools
import matplotlib.cm as cm

def save_image(diffusion_module, x_start_tokens, x_ts, x_starts, num_diffusion_steps, renderer, pgt_smpls, bev_smpls, output_fn):
    denoised_params = diffusion_module.tokenizer.split_tokens(x_start_tokens)
    denoised_smpls = diffusion_module.get_smpl(denoised_params)

    # visualize results
    max_images = 32

    output = {
        'target': pgt_smpls,  
        'bev': bev_smpls,
        'input': diffusion_module.get_smpl(
            diffusion_module.tokenizer.split_tokens(x_ts[num_diffusion_steps-1])),
        'denoised': denoised_smpls,
    }
    meshcols = {
            'target': ['light_blue1', 'light_blue6'], 
            'bev': ['light_green1', 'light_green6'],
            'input': ['light_red1', 'light_red6'],
            'denoised': ['light_yellow1', 'light_yellow6'], 
    }

    for k in np.arange(1,0,1): #[99, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1]: # [999, 750, 500, 250, 125, 75, 20, 10, 1]: #[99, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1]:
        v = x_ts[k]
        v1 = x_starts[k]
        output[k] = diffusion_module.get_smpl(diffusion_module.tokenizer.split_tokens(v))
        meshcols[k] = ['light_red1', 'light_red6']
        output[f'{k}_denoised'] = diffusion_module.get_smpl(diffusion_module.tokenizer.split_tokens(v1))
        meshcols[f'{k}_denoised'] = ['light_yellow1', 'light_yellow6']

    num_methods = int(len(output.keys()))
    view_to_row = {-20: 0, 20: 1, 270: 2} # mapping between rendering view and row index in image (per method)
    num_views = len(view_to_row.keys())
    ih, iw = renderer.ih, renderer.iw
    diffusion_module.final_image_out = np.zeros((max_images * ih, num_methods * num_views * iw, 4))
    
    # render meshes for outputs
    for idx, name in enumerate(output.keys()):
        verts_h0 = [output[name][0].vertices[[iidx]].detach() for iidx in range(max_images)]
        verts_h1 = [output[name][1].vertices[[iidx]].detach() for iidx in range(max_images)]
        diffusion_module.render_one_method(max_images, verts_h0, verts_h1, 
            diffusion_module.body_model_type, meshcols[name], 
            diffusion_module.faces_tensor, view_to_row, idx, None
        )
    # save image
    cv2.imwrite(output_fn, diffusion_module.final_image_out[...,:3])




def save_gif(renderer, diffusion_module, x_ts, num_diffusion_steps, output_fn, x_starts, 
    num_methods = 1, max_images = 4, view_to_row = {-20: 0, 20: 1, 270: 2}, 
    is_batch=False
    ):
    
    num_views = len(view_to_row.keys())
    ih, iw = renderer.ih, renderer.iw
    
    # render meshes for outputs
    images_noise, images_pred = [], []

    # render the input / start mesh 
    diffusion_module.final_image_out = np.zeros((max_images * ih, num_methods * num_views * iw, 4))
    #input_meshes = diffusion_module.get_smpl(
    #            diffusion_module.tokenizer.split_tokens(x_ts[num_diffusion_steps-1]))
    rk = list(x_starts.keys())[-1]
    input_meshes = x_starts[rk] #x_ts[num_diffusion_steps-1]
    if is_batch:
        verts_h0 = input_meshes['vertices'][:max_images,[0],:,:].detach() #[input_meshes[0].vertices[[iidx]].detach() for iidx in range(max_images)]
        verts_h1 = input_meshes['vertices'][:max_images,[1],:,:].detach() #[input_meshes[1].vertices[[iidx]].detach() for iidx in range(max_images)]        
    else:
        verts_h0 = [input_meshes[0].vertices[[iidx]].detach() for iidx in range(max_images)]
        verts_h1 = [input_meshes[1].vertices[[iidx]].detach() for iidx in range(max_images)]
    
    diffusion_module.render_one_method(max_images, verts_h0, verts_h1, 
        diffusion_module.body_model_type, ['light_blue1', 'light_blue6'], 
        diffusion_module.faces_tensor, view_to_row, 0, None
    )
    # calculate center to fix camera position
    #images.append(diffusion_module.final_image_out[...,:3].astype(np.uint8))
    if is_batch:
        centers = x_ts[rk]['vertices'][:max_images].mean((2,1))
    else:
        centers = torch.cat(
            [input_meshes[0].vertices.unsqueeze(0), input_meshes[1].vertices.unsqueeze(0)], 
        dim=0).mean((2, 0))

    # render every step in the diffusion process
    #for idx in np.arange(0,num_diffusion_steps)[::-1]:
    for kk in sorted(list(x_ts.keys()))[::-1]:
        diffusion_module.final_image_out = np.zeros((max_images * ih, num_methods * num_views * iw, 4))
        #bbodies = diffusion_module.get_smpl(diffusion_module.tokenizer.split_tokens(x_starts[idx]))
        if is_batch:
            verts_h0 = x_ts[kk]['vertices'][:max_images,[0],:,:].detach()
            verts_h1 = x_ts[kk]['vertices'][:max_images,[1],:,:].detach()
        else:
            bbodies = x_ts[kk]
            verts_h0 = [bbodies[0].vertices[[iidx]].detach() for iidx in range(max_images)]
            verts_h1 = [bbodies[1].vertices[[iidx]].detach() for iidx in range(max_images)]
        diffusion_module.render_one_method(max_images, verts_h0, verts_h1, 
            diffusion_module.body_model_type, ['light_yellow1', 'light_yellow6'], 
            diffusion_module.faces_tensor, view_to_row, 0, None, centers)
        images_noise.append(diffusion_module.final_image_out[...,:3].astype(np.uint8))

    for kk in sorted(list(x_starts.keys()))[::-1]:
        diffusion_module.final_image_out = np.zeros((max_images * ih, num_methods * num_views * iw, 4))
        #bbodies = diffusion_module.get_smpl(diffusion_module.tokenizer.split_tokens(x_starts[idx]))
        if is_batch:
            verts_h0 = x_starts[kk]['vertices'][:max_images,[0],:,:].detach()
            verts_h1 = x_starts[kk]['vertices'][:max_images,[1],:,:].detach()
        else:
            bbodies = x_starts[kk]
            verts_h0 = [bbodies[0].vertices[[iidx]].detach() for iidx in range(max_images)]
            verts_h1 = [bbodies[1].vertices[[iidx]].detach() for iidx in range(max_images)]
        diffusion_module.render_one_method(max_images, verts_h0, verts_h1, 
            diffusion_module.body_model_type, ['light_blue1', 'light_blue6'], 
            diffusion_module.faces_tensor, view_to_row, 0, None, centers)
        images_pred.append(diffusion_module.final_image_out[...,:3].astype(np.uint8))

    # add text to image
    kks = sorted(list(x_ts.keys()))[::-1]
    for idx, tt in enumerate(kks):
        cv2.putText(images_pred[idx],
            f"step {tt} (pred)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(images_noise[idx],
            f"step {tt} (noise)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    final = []
    for idx in range(len(images_pred)):
        final.append(images_noise[idx])
        final.append(images_pred[idx])
    # save gif
    imageio.mimsave(output_fn, final, format="GIF-PIL", duration=1)
    imageio.mimsave(output_fn.replace('.gif', '_noise.gif'), images_noise, format="GIF-PIL", duration=1)
    imageio.mimsave(output_fn.replace('.gif', '_pred.gif'), images_pred, format="GIF-PIL", duration=1)



def flatten(x):
    x = list(itertools.chain(*x))
    if isinstance(x[0], list):
        return flatten(x)
    return x

def create_hist_of_errors(error_dict, param_name, x_label, color_dict=None, key_filter=None, output_folder='evaluate/output'):

    # generate a colormap in form of dictionary
    if color_dict is None:
        cmap = cm.get_cmap('viridis')
        color_dict = {i: cmap(i/100) for i in range(len(error_dict.keys()))}
        
    all_data = []
    for i, (label, data) in enumerate(error_dict.items()):
        if key_filter is not None and key_filter not in label:
            continue
        flattened_list = flatten(data)
        plt.hist(flattened_list, bins=20, label=label, color=color_dict[i], alpha=0.5)
        all_data.extend(flattened_list)

    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.title("Multiple Histograms")
    plt.xlim([min(all_data), max(all_data)])
    plt.legend()
    plt.savefig(f'{output_folder}/{param_name}_{key_filter}.png')
    plt.close()

def create_hist_of_errors_gif(error_dict, param_name, x_label, color_dict=None, key_filter=None, output_folder='evaluate/output'):

    temp_dir = f'{output_folder}/temp_for_histogram_fig'
    os.makedirs(temp_dir)

    # generate a colormap in form of dictionary
    if color_dict is None:
        cmap = cm.get_cmap('viridis')
        color_dict = {i: cmap(i/100) for i in range(len(error_dict.keys()))}
    
    # find the minimum and maximum values of all data to set axis limits
    all_data = []
    for i, (label, data) in enumerate(error_dict.items()):
        if key_filter is not None and key_filter not in label:
            continue
        flattened_list = flatten(data)
        all_data.extend(flattened_list)
    min_max = [min(all_data), max(all_data)]
    ymax = 0.1 * len(flattened_list)

    # create histogram
    for i, (label, data) in enumerate(error_dict.items()):
        if key_filter is not None and key_filter not in label:
            continue
        flattened_list = flatten(data)
        plt.hist(flattened_list, bins=20, label=label, color=color_dict[i], alpha=0.5)
        plt.xlabel(x_label)
        plt.ylabel("Frequency")
        plt.title("Multiple Histograms")
        plt.xlim(min_max)
        plt.ylim([0, ymax])
        plt.legend()
        plt.savefig(f'{temp_dir}/{i:06d}.png')
        plt.close()

    # read images to list
    final = []
    for i, (label, data) in enumerate(error_dict.items()):
        if key_filter is not None and key_filter not in label:
            continue
        final.append(imageio.imread(f'{temp_dir}/{i:06d}.png'))

    # save gif
    imageio.mimsave(f'{output_folder}/{param_name}_{key_filter}.gif', final, format="GIF-PIL", duration=1)

    # delete folder temp_for_histogram_fig
    shutil.rmtree(temp_dir)

def render_images(vertices, diffusion_module, max_images, OUTPUT_FOLDER, img_prefix=''):
    """
    Render each sample and write to disk.
    vertices: [bs, 2, num_verts, 3]
    """

    centers = vertices[:max_images].mean((2,1)).mean(0)[None]
    view_to_row = {-20: 0, 20: 1, 270: 2}
    diffusion_module.final_image_out = np.zeros((1 * diffusion_module.renderer.ih, 1 * len(view_to_row.keys()) * diffusion_module.renderer.iw, 4))
    os.makedirs(f'{OUTPUT_FOLDER}/images', exist_ok=True)
    for img_idx in range(max_images):
        verts_h0 = vertices[[[img_idx]],[0],:,:].detach()
        verts_h1 = vertices[[[img_idx]],[1],:,:].detach()
        try:
            diffusion_module.render_one_method(1, verts_h0, verts_h1, 
                diffusion_module.body_model_type, ['light_blue1', 'light_blue6'], 
                diffusion_module.faces_tensor, view_to_row, 0, None, centers[[0]])
        except Exception as e:
            print(e)
            continue
        out_img = diffusion_module.final_image_out
        cv2.imwrite(f'{OUTPUT_FOLDER}/images/{img_prefix}{img_idx:04d}.png', out_img[...,:3])