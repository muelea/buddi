import subprocess 

RENDER_CFG = 'llib/visualization/scripts_vis_paper/cfg_render_highres.yaml'
#SELECTION_CFG = 'llib/visualization/scripts/paper_image_filenames_video.txt'
#SELECTION_CFG = 'llib/visualization/scripts_vis_paper/cvpr2024_images.txt'
SELECTION_CFG = 'llib/visualization/scripts_vis_paper/cvpr2024_all_images.txt'
# OUT_ROOT = '/home/lmueller/projects/HumanHumanContact/humanhumancontact/outdebug/render_highres_v06_video'
OUT_ROOT = '/ps/project/socialtouch/external_datasets/processed/FlickrCI3D_Signatures/test/renders/failure_cases'
# METHODS = ['bev', 'fit_diffprior', 'fit_pseudogt', 'fit_baseline']
RESULT_ROOT = '/is/cluster/work/lmueller2/results/HHC/optimization/cvpr2023'

METHODS = ['bev', 'fit_all_buddi_13_buddi_cond_flickrchi3dhi4d_buddi_cond_bev', 'pseudo_ground_truth', 'fit_all_buddi_13_buddi_cond_flickrchi3dhi4d_gen_buddi_cond_bev_gen', 'fit_all_heuristic_heuristic_02', 'fit_all_vae_01_vae_flickrchi3dhi4d_v2_vae_02', 'fit_pseudogt']
#METHODS = ['fit_all_buddi_13_buddi_cond_flickrchi3dhi4d_buddi_cond_bev']
base_cmd = 'python llib/visualization/scripts_vis_paper/render_dataset.py --out_root {}  --data_type flickr --data_split test --render_cfg {} --data_root {}  --body_cfg {} --method {} --selection {} --result_root {} --top --src --side --gif'

# Only save render dict 
base_cmd += ' --save_render_dict_only'

for METHOD in METHODS:
    BODY_MODEL_CFG = 'llib/visualization/scripts_vis_paper/cfg_body_model.yaml' if METHOD != 'bev' \
        else 'llib/visualization/scripts_vis_paper/cfg_body_model_smpl.yaml'
    #DATA_FOLDER = '/shared/lmueller/projects/humanhumancontact/results/optimization/final_on_flickr_ci3d_test/' \
    #    if METHOD != 'bev' else  '/home/lmueller/projects/HumanHumanContact/humanhumancontact/datasets'
    DATA_FOLDER = 'datasets'
    cmd = base_cmd.format(OUT_ROOT, RENDER_CFG, DATA_FOLDER, BODY_MODEL_CFG, METHOD, SELECTION_CFG, RESULT_ROOT)
    print(cmd)
    subprocess.call(cmd, shell=True)
