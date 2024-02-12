conda create -n hhcenv39 python=3.9 -y
conda run -n hhcenv39 --live-stream conda install -c pytorch pytorch=1.9.1 torchvision cudatoolkit=10.2 -y
conda run -n hhcenv39 --live-stream conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda run -n hhcenv39 --live-stream conda install -c bottler nvidiacub -y
conda run -n hhcenv39 --live-stream conda install pytorch3d -c pytorch3d -y
conda run -n hhcenv39 --live-stream conda install -c conda-forge tensorboard -y
conda run -n hhcenv39 --live-stream pip install opencv-python smplx scipy scikit-image loguru omegaconf ipdb einops chumpy trimesh setuptools==58.2.0
conda run -n hhcenv39 --live-stream pip install 'git+https://github.com/facebookresearch/detectron2.git' 
pip install mmcv==1.3.9 timm
pip install -v -e third-party/ViTPose/
pip install simple_romp==1.1.3

YOUR_PROJECT_FOLDER=$(pwd)
export PYTHONPATH="$YOUR_PROJECT_FOLDER/buddi"