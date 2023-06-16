## Quick start
- You need [SMPL-X](https://smpl-x.is.tue.mpg.de), [SMPL-A](https://github.com/Arthur151/ROMP#news), and [SMIL](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html).
- Note that SMPL-X, SMIL, and SMPL-A underlie different licences (not MIT)
- Download each model from the website and extract data to $SMPL_FOLDER. Then symlink $SMPL_FOLDER to essentials: `ln -s $SMPL_FOLDER $REPO_ROOT/essentials/body_models`. The output should look like this:
    ```bash
    ├── $REPO_ROOT
    │   ├── essentials
    │   │   ├── body_models
    │   │   │   ├── smil
    │   │   │   │   ├── smil_packed_info.pth
    │   │   │   │   ├── smplx_kid_template.npy
    │   │   │   ├── smplx
    │   │   │   │   ├── SMPLX_NEUTRAL.npz
    │   │   │   │   ├── SMPLX_NEUTRAL.pkl
    │   │   │   ├── smpla
    │   │   │   │   ├── SMPLA_NEUTRAL.pth
    ```

- Clone [imar vision tools](https://github.com/sminchisescu-research/imar_vision_datasets_tools.git) repo to `$REPO_ROOT`/essentials:
    ```
    git clone https://github.com/sminchisescu-research/imar_vision_datasets_tools.git essentials/imar_vision_datasets_tools
    ```
    ```bash
    ├── $REPO_ROOT
    │   ├── essentials
    │   │   ├── imar_vision_datasets_tools
    ```
    
## Essentials
- You can skip this step if you'll be using install.sh
- Dowload buddi supporting files from [here](https://www.dropbox.com/scl/fo/tfb2geh22eeprf1rnbhnp/h?dl=0&rlkey=ardm51bvjh8kuoq2lq5dtzip2) and extract them to `$REPO_ROOT`/essentials
    ```bash
    ├── $REPO_ROOT
    │   ├── essentials
    │   │   ├── buddi
    │   │   │   ├── buddi_checkpoint_v01.pt
    │   │   │   ├── config_v01.yaml
    │   │   ├── priors
    │   │   │   ├── gmm_08.pkl
    │   │   ├── contact
    │   │   │   ├── flickrci3ds_r75_rid_to_smplx_vid.pkl
    │   │   ├── body_models
    │   │   │   ├── smil
    │   │   │   │   ├── smplx_kid_template.npy
    │   │   ├── body_model_utils
    │   │   │   ├── lowres_smplx.pkl
    │   │   │   ├── smplx_faces.pt
    │   │   │   ├── smpl_to_smplx.pkl
    │   │   │   ├── smplx_inner_mouth_bounds.pkl
    ```

## Installation
 - Install virtual environment with python packages
    ```
    # Eventually update conda 
    # conda update conda
    conda create -n hhcenv39 python=3.9
    conda activate hhcenv39
    conda install -c pytorch pytorch=1.9.1 torchvision cudatoolkit=10.2
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -c bottler nvidiacub
    conda install pytorch3d -c pytorch3d
    conda install -c conda-forge tensorboard
    pip install opencv-python smplx scipy scikit-image loguru omegaconf ipdb einops chumpy trimesh setuptools==58.2.0
    ```

- Eventually, you need to add environment variables in shell or to you .bashrc
    ```
    YOUR_PROJECT_FOLDER=$(pwd)
    export PYTHONPATH="$YOUR_PROJECT_FOLDER/humanhumancontact"
    ```

- Install conda environment and download the supporting files via shell script
    ```bash
    scripts/install.sh
    ```