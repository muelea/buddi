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
