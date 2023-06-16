conda create -n hhcenv39 python=3.9 -y
conda run -n hhcenv39 --live-stream conda install -c pytorch pytorch=1.9.1 torchvision cudatoolkit=10.2 -y
conda run -n hhcenv39 --live-stream conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda run -n hhcenv39 --live-stream conda install -c bottler nvidiacub -y
conda run -n hhcenv39 --live-stream conda install pytorch3d -c pytorch3d -y
conda run -n hhcenv39 --live-stream conda install -c conda-forge tensorboard -y
conda run -n hhcenv39 --live-stream pip install opencv-python smplx scipy scikit-image loguru omegaconf ipdb einops chumpy trimesh setuptools==58.2.0

PWD=$(pwd)
export PYTHONPATH="$PWD"
export BUDDI_HOME="$PWD"
#export DATASETS_HOME="$PWD"
#export ESSENTIALS_HOME="$PWD/essentials/"


zip_url="https://www.dropbox.com/s/37nlo2opphpjc74/essentials.zip"
zip_file="essentials.zip"
extract_folder="essentials"

# Download essentials.zip
wget "$zip_url" -O "$zip_file"

# Extract contents to data/ folder
unzip "$zip_file" #-d "$extract_folder"

# Delete the ZIP file
rm "$zip_file"
rm -r "__MACOSX/"