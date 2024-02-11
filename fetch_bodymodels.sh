#### Download Body Models ####

cd essentials/body_models

########### SMPL-X ###########
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Login to SMPL-X Website
# If you don't have an account, create one here: https://smpl-x.is.tue.mpg.de/
read -p "Username (SMPl-X Website - https://smpl-x.is.tue.mpg.de/):" username
read -p "Password (SMPl-X Website - https://smpl-x.is.tue.mpg.de/):" -s password

username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip&resume=1' -O 'models_smplx_v1_1.zip' --no-check-certificate --continue
unzip models_smplx_v1_1.zip
mv models/smplx .
rm models_smplx_v1_1.zip
rm -r models



########### SMPL ###########
echo "Downloading SMPL Models"

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

username=$(urle $username)
password=$(urle $password)

# Login to SMPL Website
# If you don't have an account, create one here: https://smpl.is.tue.mpg.de/
read -p "Username (SMPL Website - https://smpl.is.tue.mpg.de/):" username
read -p "Password (SMPL Website - https://smpl.is.tue.mpg.de/):" -s password

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip&resume=1' -O 'models_smpl.zip' --no-check-certificate --continue
unzip models_smpl.zip
mkdir smpl
cp 'SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl' smpl/SMPL_NEUTRAL.pkl
rm -r SMPL_python_v.1.1.0
rm models_smpl.zip




########## SMIL ###########
echo "Downloading SMIL Models"
mkdir smil_temp && cd smil_temp
wget 'https://obj-web.iosb.fraunhofer.de/content/sensornetze/bewegungsanalyse/smil.zip' -O 'smil_web.zip' --no-check-certificate --continue
unzip smil_web.zip
rm smil_web.zip
mv smil/smil_web.pkl ../smil/
cd ..