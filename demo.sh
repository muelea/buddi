# This script runs ViTPose and BEV on your images and then starts the optimization with BUDDI.
# If you have OpenPose installed, you can also run OpenPose on your images as well 
# and set the datasets.demo.openpose_folder to the folder where the OpenPose keypoints are stored.

# Run ViTPose on your image
echo "Running ViTPose on your image"

mkdir demo/data/FlickrCI3D_Signatures/demo/vitpose_live
python llib/utils/keypoints/vitpose_model.py --image_folder demo/data/FlickrCI3D_Signatures/demo/images_live --out_folder demo/data/FlickrCI3D_Signatures/demo/vitpose_live



# Run BEV on your images 
echo "Running BEV on your images"

mkdir demo/data/FlickrCI3D_Signatures/demo/bev_live
for image in demo/data/FlickrCI3D_Signatures/demo/images_live/*; do
    # get only image name
    image_name=$(basename $image)
    bev -i $image -o demo/data/FlickrCI3D_Signatures/demo/bev_live/$image_name
done


# Run OpenPose on your images 
echo "Not Running OpenPose on your images"
# We don't have OpenPose installed in this repo and run the demo on random images
# with keypoints detected by ViTPose (wholebody) model only. This more recent model 
# works quite well and is a good alternative to OpenPose.

# If you wish to run the original version with ViTPose (core body) + OpenPose (wholebody) keypoints, 
# you can install OpenPose from here https://github.com/CMU-Perceptual-Computing-Lab/openpose
# and use this command: ./build/examples/openpose/openpose.bin --image_dir demo/data/FlickrCI3D_Signatures/demo/images --face --hand
# I keep it in openpose/examples/tutorial_api_python/ folder
# Here is a script I use for OpenPose: llib/utils/keypoints/run_openpose_folder.py
# OpenPose Docs: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md

# To pass OpenPose keypoints to the optimization script set
# datasets.demo.openpose_folder to the folder where the OpenPose keypoints are stored.

# Run Optimization with BUDDI
echo "Running Optimization with BUDDI"

# Run Optimization with BUDDI
# For live demo we set the openpose_folder to none
# If you have OpenPose installed, you can pass the openpose keypoints folder to the openpose_folder
# to test without demo images set datasets.demo.openpose_folder=keypoints/keypoints
python llib/methods/hhcs_optimization/main.py --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml --exp-opts logging.base_folder=demo/optimization/buddi_cond_bev_demo_live datasets.train_names=['demo'] datasets.train_composition=[1.0] datasets.demo.original_data_folder=demo/data/FlickrCI3D_Signatures/demo datasets.demo.image_folder=images_live datasets.demo.bev_folder=bev_live datasets.demo.vitpose_folder=vitpose_live datasets.demo.openpose_folder=none model.optimization.pretrained_diffusion_model_ckpt=essentials/buddi/buddi_cond_bev.pt model.optimization.pretrained_diffusion_model_cfg=essentials/buddi/buddi_cond_bev.yaml logging.run=fit_buddi_cond_bev_flickrci3ds