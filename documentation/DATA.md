## Datasets
We use two datasets in this project, CHI3D and FlickrCI3D Signatures, which you can download from [this website](https://ci3d.imar.ro/download). I keep them in a seperate folder named $ORIG_DATASETS_FOLDER. Then download additional resources like keypoints, BEV etc from our website. Save them in a folder named $PROCESSED_DATASETS_FOLDER. I recommend symlinks to this $REPO/datasets so that you don't need to change paths in the config file:

```
ln -s $ORIG_DATASETS_FOLDER $REPO_ROOT/datasets/original
ln -s $PROCESSED_DATASETS_FOLDER $REPO_ROOT/datasets/processed
```


### Download original CHI3D and FlickrCI3D Signatures
- Download CHI3D and FlickrCI3D Signatures from [this website](https://ci3d.imar.ro/download) to $ORIG_DATASETS_FOLDER/CHI3D and $ORIG_DATASETS_FOLDER/FlickrCI3D_Signatures, respectively. 
- Extract train and test archives (e.g. on ubuntu: `tar xvzf chi3d_train.tar.gz`). 
- Clone imar vision tools repo to $ORIG_DATASETS_FOLDER: `cd $ORIG_DATASETS_FOLDER && git clone https://github.com/sminchisescu-research/imar_vision_datasets_tools.git`

The folderstructure should look like this:

```bash
├── $ORIG_DATASETS_FOLDER
│   ├── imar_vision_datasets_tools
│   ├── CHI3D
│   │   ├── chi3d_info.json
│   │   ├── chi3d_template.json
│   │   ├── train
│   │   ├── test
│   ├── FlickrCI3D_Signatures
│   │   ├── train
│   │   ├── test
```


### Download additional resources for CHI3D and FlickrCI3D Signatures
- Download auxiliary data like BEV, keypoints, etc. and place it in datasets/processed

Final sturcture should look like this
```bash
├── $PROCESSED_DATASETS_FOLDER
│   ├── CHI3D
│   │   ├── train
│   │   ├── test
│   ├── FlickrCI3D_Signatures
```

### Extract frames from videos (for CHI3D) 
- For CHI3D you can use `datasets/scripts/CHI3D/extract_frames.py` to extract video frames. Here we save the images in the processed dataset folder using this format f'{output_folder}/{split}/images/{action}_{frame_index:06d}_{cam}.jpg'
`python datasets/scripts/CHI3D/extract_frames.py --input_folder $ORIG_DATASETS_FOLDER/CHI3D --output_folder $PROCESSED_DATASETS_FOLDER/CHI3D --sequence all`
