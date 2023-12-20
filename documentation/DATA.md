# Datasets

## Overview
We use three datasets for this project, CHI3D and FlickrCI3D Signatures, which you can download 
from [this website](https://ci3d.imar.ro/download), and Hi4D which you can download [here](https://yifeiyin04.github.io/Hi4D/#dataset). 
Keep the original data in a seperate folder named `$ORIG_DATASETS_FOLDER`. Then download our auxiliary data (OpenPose, ViTPose, BEV estimates etc.)
for each dataset and save them in `$PROCESSED_DATASETS_FOLDER`. Create symlinks to this repo, `$BUDDI_ROOT`, so that you don't need to change 
paths in the config file:

```bash
ln -s $ORIG_DATASETS_FOLDER $BUDDI_ROOT/datasets/original
ln -s $PROCESSED_DATASETS_FOLDER $BUDDI_ROOT/datasets/processed
```

------------
## FlickrCI3D
------------
### Original data (images and ground-truth contact map annotations)
- Download FlickrCI3D Signatures Training and Testing Set from [this website](https://ci3d.imar.ro/download) and extract archives to `$ORIG_DATASETS_FOLDER`.
    ```bash
    ├── $ORIG_DATASETS_FOLDER
    │   ├── FlickrCI3D_Signatures
    │   │   ├── train
    │   │   │   ├── images
    │   │   │   ├── interaction_contact_signature.json    
    │   │   ├── test
    │   │   │   ├── images
    │   │   │   ├── interaction_contact_signature.json   
    ```

### Auxiliary data (pseudo-ground truth fits)
- Download our pseudo ground-truth fits for FlickrCI3D and auxiliary data like BEV, keypoints, etc. from [here](url.url).
- Extract data to `$PROCESSED_DATASETS_FOLDER`
    ```bash
    ├── $PROCESSED_DATASETS_FOLDER
    │   ├── FlickrCI3D_Signatures
    │   │   ├── train 
    │   │   |   ├── train_val_split.npz 
    │   │   |   ├── processed.pkl
    │   │   |   ├── processed_pseudogt_fits.pkl
    │   │   ├── test
    │   │   |   ├── train_val_split.npz 
    │   │   |   ├── processed.pkl
    │   │   |   ├── processed_pseudogt_fits.pkl
    ```

----------
## Hi4D
------------
### Original data (videos, images, ground-truth SMPL etc.)
- Register to get acces to Hi4D data. They will provide a personal URL for you to download Hi4D. 
- Here is a script I found useful. Place the script in $ORIG_DATASETS_FOLDER/Hi4D. Running it will list all tar.gz files from the Hi4D website in files.txt.
    ```python
    import requests
    from bs4 import BeautifulSoup

    PERSONAL_Hi4D_URL = "your person URL for Hi4D"  # Replace with your desired URL

    def list_tar_gz_files(url):
        base_url = "https://hi4d.ait.ethz.ch"
        try:
            # Send an HTTP GET request to the provided URL
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all anchor (a) tags in the HTML with href ending in .tar.gz
            anchor_tags = soup.find_all('a', href=lambda href: href.endswith('.tar.gz'))

            # Prepare the content for files.txt
            file_list = []
            for anchor in anchor_tags:
                href = anchor['href']
                output_filename = href.split('/')[-1]
                file_list.append(f"{base_url}/{href} -O {output_filename}")

            # Save the content to files.txt
            with open('files.txt', 'w') as file:
                file.write('\n'.join(file_list))

            print("files.txt created successfully with the list of .tar.gz files.")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

    # Example usage:
    list_tar_gz_files(PERSONAL_Hi4D_URL)
    ```
- Now you can download all Hi4D files from the command line via: 
    ```bash
    while read FOO; do wget $FOO; done < files.txt
    ```

- Finally, extract the tar.gz data. Some paris have two folder (e.g. pair00_1 and pair00_2). This data should be extracted to the same folder like this: 
    ```bash
    pair00
    --content_pair00_1
    --content_pair00_2
    ```

    ```bash
    #for pp in "00" "02" "13" "15" "17" "18" "21" "23" "27" "37"; do mkdir pair$pp && mv pair$pp\_1/* pair$pp/ && mv pair$pp\_2/* pair$pp/ && rm -r pair$pp\_1 pair$pp\_2; done
    for pp in "00" "02" "13" "15" "17" "18" "21" "23" "27" "37"; do mkdir pair$pp && tar xf pair$pp\_1.tar.gz -C pair$pp --strip-components 1 && tar xf pair$pp\_2.tar.gz -C pair$pp --strip-components 1; done
    for pp in "01" "09" "10" "12" "14" "16" "19" "22" "28" "32"; do tar xf pair$pp.tar.gz; done

    ```

    You can also remove a the leftover mp4 file in pair02/talk02:
    ```
    rm pair02/talk02/images/88/talk02_cam88.mp4 pair13/highfive13/images/64/highfive13_cam64.mp4 pair02/talk02/images/52/talk02_cam52.mp4 pair16/jump16/images/16/jump16_cam16.mp4 pair16/jump16/images/40/jump16_cam40.mp4 pair19/piggyback19/images/4/piggy19_cam4.mp4 pair19/piggyback19/images/76/piggy19_cam76.mp4 pair22/kiss22/images/4/kiss22_cam4.mp4
    ```

### Auxiliary data (SMPL-X parameters)
- Download auxiliary data from [here](url.url) and extract data to `$PROCESSED_DATASETS_FOLDER`
    ```bash
    ├── $PROCESSED_DATASETS_FOLDER
    │   ├── Hi4D
    │   │   ├── train_val_split.npz 
    │   │   ├── processed.pkl
    │   │   ├── processed_single_camera.pkl
    ```

    The data in processed.pkl contains the ground-truth 3D SMPL and SMPL-X fits and in data[<PAIR>][<ACTION>]['image_data'] the BEV estimate and keypoint detections for each image.

- Single steps to create processed.pkl. Skip if you do not want to preprocess Hi4D. For training BUDDI you don't need this either.
    1) Ground-truth Hi4D parameters are in SMPL format, but we convert SMPL to SMPL-X for this project. We first save ground-truth SMPL
    meshes of each person as obj files in $PROCESSED_DATASETS_FOLDER/Hi4D/smpl, e.g. Hi4D/pair00/dance00/smpl/000048_0.obj.
        ```bash
        # install this repo: https://github.com/vchoutas/smplx
        cd $TOOLS_FOLDER/smplx
        source .venv/smplx/bin/activate
        python -m transfer_model --exp-cfg config_files/smpl2smplx.yaml

        # To run on folder
        python run_on_folder $PROCESSED_DATASETS_FOLDER/Hi4D/smpl config_files/smpl2smplx.yaml
        ```

    2) Run ViTPose to get 2D keypoints
        ```bash
        # install this repo: https://github.com/ViTAE-Transformer/ViTPose
        cd $TOOLS_FOLDER/ViTPose
        source .venv/vitpose/bin/activate
        export DS_ROOT='$ORIG_DATASETS_FOLDER/FlickrCI3D_Signatures/test'
        python demo/top_down_img_demo_with_mmdet.py     demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py     https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth     configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth     --img-root $DS_ROOT/images/  --out-img-root $DS_ROOT/keypoints/vitposeplus_images --out-res-root $DS_ROOT/keypoints/vitposeplus
        ```

    3) Processing scrip to created processed.pkl
        ``` bash
        cd $REPO_ROOT
        python llib/data/preprocess/process_hi4d.py
        ```

----------
## CHI3D
------------
### Original data (videos and mocap)
- Required to train BUDDI, but can be ignored if you want to train BUDDI on Flickr PGT only.
- Download CHI3D Signatures from [this website](https://ci3d.imar.ro/download) and extract archives to `$ORIG_DATASETS_FOLDER`.
- If not done already, clone [imar vision tools](https://github.com/sminchisescu-research/imar_vision_datasets_tools.git) repo to `$ORIG_DATASETS_FOLDER`.
    ```bash
    ├── $ORIG_DATASETS_FOLDER
    │   ├── CHI3D
    │   │   ├── chi3d_info.json
    │   │   ├── chi3d_template.json
    │   │   ├── train
    │   │   ├── test
    ```

### Auxiliary data (BEV, OpenPose, VitPose)
- Download auxiliary data from [here](url.url) and extract data to `$PROCESSED_DATASETS_FOLDER`
    ```bash
    ├── $PROCESSED_DATASETS_FOLDER
    │   ├── CHI3D
    │   │   ├── train_val_split.npz 
    │   │   ├── images_contact_processed.pkl 
    ```

- To extract frames from CHI3D videos via `extract_frames.py` (this will take several hours).
    You can skip this step since the images and not used during training. In this case, you can now run the demo training code. 
    The data loader will use an empty image and you won't see the original image. Extract_frames.py will save the images to
    - {output_folder}/train/{subject}/images/{action}\_{frame_index:06d}\_{cam} for training, and
    - {output_folder}/test/{subject}/images/{action}\_{frame_index:06d} for test.
    ```bash
    python datasets/scripts/CHI3D/extract_frames.py --input_folder $ORIG_DATASETS_FOLDER/CHI3D --output_folder $PROCESSED_DATASETS_FOLDER/CHI3D --sequence all
    ```
    ```bash
    ├── $PROCESSED_DATASETS_FOLDER
    │   ├── CHI3D
    │   │   ├── train
    │   │   │   ├── subject
    │   │   │   │   ├── images
    │   │   ├── test
    │   │   │   ├── subject
    │   │   │   │   ├── images
    ```
