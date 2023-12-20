import subprocess
import os
import argparse


def run_sampling(base_folder, run_folder, checkpoint_epoch, config_string):

    run_dir = f"{base_folder}/{run_folder}"

    # find checkpoint path for selected epoch
    checkpoint_names = os.listdir(f"{run_dir}/checkpoints")
    for x in checkpoint_names:
        print(x, x.split("__")[-3])
        if int(x.split("__")[-3]) == int(checkpoint_epoch):
            checkpoint_name = x
            break

    # Construct the shell command
    cmd = f"python llib/methods/hhc_diffusion/evaluation/sample.py \
        --exp-cfg {run_dir}/config.yaml \
        --output-folder {run_dir}/samples/checkpoint_{checkpoint_epoch} \
        --checkpoint-name {run_dir}/checkpoints/{checkpoint_name} \
        --max-images-render=100 --num-samples 100 --save-vis {config_string} \
        --log-steps=100"

    print('----- RUNNING CMD -------')
    print(cmd)
    
    # Run the shell command
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, required=True,
                        help = "Base folder where all the runs are stored")
    parser.add_argument("--run_folders", nargs='+', 
                        default=["00_buddi_flickrchi3dhi4d", "01_buddi_flickrchi3d",  "02_buddi_chi3d",  "02_buddi_hi4d"],
                        help="List of run folders to walk through and do the sampling")
    parser.add_argument("--checkpoint_epoch", type=str, default='899',
                        help="The epoch for which the checkpoint is to be used")
    args = parser.parse_args()

    # try different sampling strategies
    config_strings = ["--max-t -1", "--max-t 1000 --skip-steps 10", "--max-t -2"]
    config_strings = ["--max-t 1000 --skip-steps 10", "--max-t 1000 --skip-steps 5", "--max-t 1000 --skip-steps 2"]
    config_strings = ["--max-t 1000 --skip-steps 10 --eta 0.0"]

    # walk through all the run folders and do the samplingß
    for rf in args.run_folders:
        for cf in config_strings:
            run_sampling(args.base_folder, rf, args.checkpoint_epoch, cf)
