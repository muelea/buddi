conda create -n hhcenv39 python=3.9 -y
conda run -n hhcenv39 --live-stream conda install -c pytorch pytorch=1.9.1 torchvision cudatoolkit=10.2 -y
conda run -n hhcenv39 --live-stream conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda run -n hhcenv39 --live-stream conda install -c bottler nvidiacub -y
conda run -n hhcenv39 --live-stream conda install pytorch3d -c pytorch3d -y
conda run -n hhcenv39 --live-stream conda install -c conda-forge tensorboard -y
conda run -n hhcenv39 --live-stream pip install opencv-python smplx scipy scikit-image loguru omegaconf ipdb einops chumpy trimesh setuptools==58.2.0
conda run -n hhcenv39 --live-stream pip install 'git+https://github.com/facebookresearch/detectron2.git' mmcv==1.3.9 timm
pip install -v -e third-party/ViTPose/
pip install simple_romp==1.1.3

PWD=$(pwd)
export PYTHONPATH="$PWD"
export BUDDI_HOME="$PWD"
#export DATASETS_HOME="$PWD"
#export ESSENTIALS_HOME="$PWD/essentials/"

# zip_url="https://www.dropbox.com/s/37nlo2opphpjc74/essentials.zip" # Old essentials file with V1 model
zip_url="https://www.dropbox.com/scl/fi/jn3r1syak62g7djr0q06d/essentials_new.zip"
zip_file="essentials.zip"
extract_folder="essentials"

# Download essentials.zip
wget "$zip_url" -O "$zip_file"  --no-check-certificate --continue

# Extract contents to data/ folder
unzip "$zip_file" #-d "$extract_folder"

# Delete the ZIP file
rm "$zip_file"
rm -r "__MACOSX/"

# link imar vision tools to essentials 
ln -s $PWD/third-party/imar_vision_datasets_tools $PWD/essentials/imar_vision_datasets_tools
