## Installation
#### Virtual environment
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

#### Environment variables
First add the folder containing this repo as environment variable:
```
export YOUR_PROJECT_FOLDER=[YOUR_PROJECT_FOLDER]
````

```
export PYTHONPATH="$YOUR_PROJECT_FOLDER/HumanHumanContact/humanhumancontact"
export HUMANHUMANCONTACT_HOME="$YOUR_PROJECT_FOLDER/HumanHumanContact/humanhumancontact"
export DATASETS_HOME="$YOUR_PROJECT_FOLDER/HumanHumanContact/humanhumancontact"
export ESSENTIALS_HOME="$YOUR_PROJECT_FOLDER/HumanHumanContact/humanhumancontact/essentials/"
```
It's better to add the environment vairables directly to your .bashrc / .profile / ... files.

#### Essentials
```bash
├── humanhumancontact
│   ├── essentials
│   │   ├── body_models
│   │   │   ├── smpl
│   │   │   ├── smplx
│   │   ├── spin
│   │   │   ├── smpl_mean_params.npz
│   │   ├── priors
│   │   │   ├── gmm_08.pkl
│   │   ├── contact
│   │   │   ├── flickrci3ds_r75_rid_to_smplx_vid.pkl
│   │   ├── body_model_utils
│   │   │   ├── smpl_faces.pt
│   │   │   ├── smplx_faces.pt
│   │   │   ├── smplx_inner_mouth_bounds.pkl
```

You can symlink these files:
```
ln -s /shared/lmueller/essentials essentials
```