# Datasets

## Overview
We use two datasets for this project, CHI3D and FlickrCI3D Signatures, which you can download from [this website](https://ci3d.imar.ro/download). Keep them in a seperate folder named `$ORIG_DATASETS_FOLDER`. Then download additional resources like keypoints, BEV, and our pseudo ground-truth fits and save them in `$PROCESSED_DATASETS_FOLDER`. Create symlinks to this repo, `$BUDDI_ROOT`, so that you don't need to change paths in the config file:

```bash
ln -s $ORIG_DATASETS_FOLDER $BUDDI_ROOT/datasets/original
ln -s $PROCESSED_DATASETS_FOLDER $BUDDI_ROOT/datasets/processed
```

------------
## FlickrCI3D
### Pseudo-ground truth fits
- Download our pseudo ground-truth fits for FlickrCI3D and auxiliary data like BEV, keypoints, etc. from [here](url.url).
- Extract data to `$PROCESSED_DATASETS_FOLDER`
    ```bash
    ├── $PROCESSED_DATASETS_FOLDER
    │   ├── FlickrCI3D_Signature
    │   │   ├── train_val_split.npz 
    │   │   ├── train_pgt.pkl
    │   │   ├── val_pgt.pkl
    ```

### Images and contact maps
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
### Merge datasets 
- Merge PGT fits with GT contact map annotations 
    ```
    python datasets/scripts/FlickrCI3D/merge_pgt_and_gt.py
    ```
    Afterwards you should see:
    ```bash
    ├── $PROCESSED_DATASETS_FOLDER
    │   ├── FlickrCI3D_Signature
    │   │   ├── train_val_split.npz 
    │   │   ├── train_pgt.pkl
    │   │   ├── train.pkl
    │   │   ├── val_pgt.pkl
    │   │   ├── val.pkl
    ``` 

----------
## CHI3D
### Videos and mocap
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

### Data processing (optional)
- You can skip this step since the images and not used during training. In this case, you can now run the demo training code. The data loader will use an empty image and you won't see the original image. 
- To extract frames from CHI3D videos via `extract_frames.py` (this will take several hours). Here we save the images to
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