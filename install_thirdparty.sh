# Install BEV / ROMP

PWD=$(pwd)

cd  third-party/ROMP
wget "https://github.com/Arthur151/ROMP/releases/download/V2.0/smpl_model_data.zip"
unzip "smpl_model_data.zip"

cd ../../
cp essentials/body_models/smpl/SMPL_NEUTRAL.pkl third-party/ROMP/smpl_model_data/

mkdir third-party/ROMP/smpl_model_data/smil
cp essentials/body_models/smil/smil_web.pkl third-party/ROMP/smpl_model_data/smil/

romp.prepare_smpl -source_dir=$PWD/third-party/ROMP/smpl_model_data
bev.prepare_smil -source_dir=$PWD/third-party/ROMP/smpl_model_data

mkdir essentials/body_models/smpla
cp ~/.romp/SMPLA_NEUTRAL.pth essentials/body_models/smpla/
cp ~/.romp/smil_packed_info.pth essentials/body_models/smil/



# Install ViTPose 
mkdir essentials/vitpose
cd essentials/vitpose
wget "https://www.dropbox.com/scl/fi/j1j5btsb2w2wi55op9h29/wholebody.pth?rlkey=xwhtomp1d7h6xlppunbzshsfe&dl=0" --no-check-certificate --continue
mv 'wholebody.pth?rlkey=xwhtomp1d7h6xlppunbzshsfe&dl=0' wholebody.pth