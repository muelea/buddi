## Quick start

These scripts do everything for you (installation, model/data download and demo). If you run into errors,
check out the [colab](https://colab.research.google.com/drive/1P7x2gY_VuFz5yjZHTRfEBzsEmA-JYGlE?usp=sharing) example.

```
# install conda environment
./install_conda_env.sh

# download essentials and models
./fetch_data.sh

# download body models (SMPL-X, SMPL, SMIL). The script will ask for you username
# and password for the SMPL-X and SMPL website. If you don't have an account, please
# register under https://smpl-x.is.tue.mpg.de/ and https://smpl.is.tue.mpg.de/.
./fetch_bodymodels.sh

# Install BEV and ViTPose and convert body models to BEV format 
./install_thirdparty.sh

# Run optimization with BUDDI on your own images
# We have some internet images in [this](./demo/data/FlickrCI3D_Signatures/demo/images_live) folder.
# The script will first run BEV and ViTPose and then start the optimization with BUDDI.
# To run the demo with OpenPose on top, please read the comments in demo.sh
./demo.sh
```

## Install Conda Environment (install_conda_env.sh)
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
    conda run -n hhcenv39 --live-stream pip install 'git+https://github.com/facebookresearch/detectron2.git' 
    pip install mmcv==1.3.9 timm
    pip install -v -e third-party/ViTPose/
    pip install simple_romp==1.1.3
    ```

- Eventually, you need to add environment variables in shell or to you .bashrc
    ```
    YOUR_PROJECT_FOLDER=$(pwd)
    export PYTHONPATH="$YOUR_PROJECT_FOLDER/buddi"
    ```

- Install conda environment and download the supporting files via shell script
    ```bash
    scripts/install_conda_env.sh
    ```

## Download Essentials and Models (fetch_data.sh)
- You can skip this step if you'll be using install.sh
- Dowload buddi supporting files from [here](https://drive.google.com/uc?id=16eYddIxKPaZU-PjrH1x0Fgsen-x0ip3f) and extract them to `$REPO_ROOT`/essentials
    ```bash
    ├── $REPO_ROOT
    │   ├── essentials
    │   │   ├── buddi
    │   │   │   ├── buddi_unconditional.pt
    │   │   │   ├── buddi_unconditional.yaml
    │   │   │   ├── buddi_cond_bev.pt
    │   │   │   ├── buddi_cond_bev.yaml
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

- Link [imar vision tools](https://github.com/sminchisescu-research/imar_vision_datasets_tools.git) repo to `$REPO_ROOT`/essentials:
    ```
    ln -s $REPO_ROOT/third-party/imar_vision_datasets_tools $REPO_ROOT/essentials/imar_vision_datasets_tools
    ```
    ```bash
    ├── $REPO_ROOT
    │   ├── essentials
    │   │   ├── imar_vision_datasets_tools
    ```

- Bash script to download essentials and Link imar vision tools repo
    ```bash
    scripts/fetch_data.sh
    ```

## Download Body Models (fetch_bodymodels.sh)
- You need [SMPL-X](https://smpl-x.is.tue.mpg.de), [SMPL](https://smpl-x.is.tue.mpg.de), [SMPL-A](https://github.com/Arthur151/ROMP#news), [SMIL](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html). The SMIL-X Kid Template can be found on tha [AGORA](https://agora.is.tue.mpg.de) website.
- Note that SMPL-X, SMIL, and SMPL-A underlie different licences (not MIT)
- Download each model from the website and extract data to $SMPL_FOLDER. Then symlink $SMPL_FOLDER to essentials: `ln -s $SMPL_FOLDER $REPO_ROOT/essentials/body_models`. 

- Then run follow the instructions here to generate the body model files used [BEV](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md#installation).
    ```
    # generate SMPLA_NEUTRAL.pth and smil_packed_info.pth
    romp.prepare_smpl -source_dir=/path/to/smpl_model_data
    bev.prepare_smil -source_dir=/path/to/smpl_model_data
    ```

- The output should look like this:
    ```bash
    ├── $REPO_ROOT
    │   ├── essentials
    │   │   ├── body_models
    │   │   │   ├── smil
    │   │   │   |   ├── smil_web.pkl
    │   │   │   │   ├── smil_packed_info.pth
    │   │   │   │   ├── smplx_kid_template.npy
    │   │   │   ├── smpl
    │   │   │   │   ├── SMPL_NEUTRAL.pkl
    │   │   │   ├── smplx
    │   │   │   │   ├── SMPLX_NEUTRAL.npz
    │   │   │   │   ├── SMPLX_NEUTRAL.pkl
    │   │   │   ├── smpla
    │   │   │   │   ├── SMPLA_NEUTRAL.pth
    ```


- Bash script to download body models. You need an account on the websites of
[SMPL-X](https://smpl-x.is.tue.mpg.de) and [SMPL](https://smpl.is.tue.mpg.de/).
If you don't have an account, create one before you run `fetch_bodymodels.sh`; 
the scirpt will ask for your credentials.
    ```bash
    scripts/fetch_bodymodels.sh
    ```