import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass

# Features and annotations a dataset can have. False by default, set all value
# to true dataset class if applicable.
@dataclass
class DatasetFeatures():
    is_itw: bool = False # image was taken in-the-wild
    
    has_dhhc_class: bool = False #discrete human-human contact class (contact, no contact, unclear)
    has_dhhc_sig: bool = False #discrete human-human contact signature
    
    has_dsc_sig: bool = False #discrete self-contact class (contact, no contact, unclear)
    has_dsc_class: bool = False #discrete self-contact signature
    
    has_gt_kpts: bool = False #ground truth 2d keypoints
    has_op_kpts: bool = False #detected openpose 2d keypoints
    
    has_gt_joints: bool = False # has ground truth 3d joints
    
    has_gender: bool = False # if dataset has gender label

    has_gt_smpl_shape: bool = False # has only ground truth smpl shape
    has_gt_smpl_pose: bool = False # has only ground truth smpl pose
    has_pgt_smpl_shape: bool = False # has pseudo ground truth smpl shape, e.g. mtp or curated fits
    has_pgt_smpl_pose: bool = False # has pseudo ground truth smpl pose, e.g. mtp or curated fits


############################# DATASETS ############################

@dataclass
class FlickrCI3D_Signatures:
    body_model_path: str = 'essentials/body_models'
    original_data_folder: str = 'datasets/original/FlickrCI3D_Signatures'
    imar_vision_datasets_tools_folder: str = 'essentials/imar_vision_datasets_tools' #'datasets/original'
    processed_data_folder: str = 'datasets/processed/FlickrCI3D_Signatures'
    max_count_regions_in_contact: int = 25
    number_of_regions: int = 75
    image_folder: str = 'images'
    bev_folder: str = 'bev'
    openpose_folder: str = 'keypoints/keypoints'
    vitpose_folder: str = 'vitpose'
    vitdet_folder: str = 'vitdet'
    image_format: str = 'png'
    pseudogt_folder: str = 'pseudogt/summaries'
    overfit: bool = False
    overfit_num_samples: int = 12
    adult_only: bool = False
    child_only: bool = False
    features: DatasetFeatures = DatasetFeatures(
        is_itw = True, 
        has_dhhc_sig = True,
        has_op_kpts = True,
    )

@dataclass
class FlickrCI3D_Classification:
    original_data_folder: str = 'datasets/original/FlickrCI3D_Classification'
    imar_vision_datasets_tools_folder: str = 'essentials/imar_vision_datasets_tools' #'datasets/original'
    processed_data_folder: str = 'datasets/processed/FlickrCI3D_Classification'
    image_folder: str = 'images'
    image_format: str = 'png'
    bev_folder: str = 'bev'
    features: DatasetFeatures = DatasetFeatures(
        is_itw = True,
        has_dhhc_class = True,
        has_op_kpts = True
    )
    overfit: bool = False
    overfit_num_samples: int = 64

@dataclass
class FlickrCI3D_SignaturesDownstream:
    body_model_path: str = 'essentials/body_models'
    original_data_folder: str = 'datasets/original/FlickrCI3D_Signatures'
    imar_vision_datasets_tools_folder: str = 'essentials/imar_vision_datasets_tools' #'datasets/original'
    processed_data_folder: str = 'datasets/processed/FlickrCI3D_Signatures'
    max_count_regions_in_contact: int = 25
    number_of_regions: int = 75
    image_folder: str = 'images'
    bev_folder: str = 'bev'
    openpose_folder: str = 'keypoints/keypoints'
    vitpose_folder: str = 'vitpose'
    vitdet_folder: str = 'vitdet'
    image_format: str = 'png'
    pseudogt_folder: str = 'pseudogt/summaries'
    overfit: bool = False
    overfit_num_samples: int = 12
    init_pose_from_bev: bool = False
    features: DatasetFeatures = DatasetFeatures(
        is_itw = True, 
        has_dhhc_sig = True,
        has_op_kpts = True,
    )

@dataclass
class CHI3D:
    original_data_folder: str  = 'datasets/original/CHI3D'
    processed_data_folder: str = 'datasets/processed/CHI3D'
    imar_vision_datasets_tools_folder: str = 'essentials/imar_vision_datasets_tools' #str = 'datasets/original'
    max_count_regions_in_contact: int = 25
    number_of_regions: int = 75
    image_folder: str = 'images'
    bev_folder: str = 'bev'
    openpose_folder: str = 'keypoints/keypoints'
    vitpose_folder: str = 'vitpose'
    vitdet_folder: str = 'vitdet'
    image_format: str = 'png'
    pseudogt_folder: str = 'pseudogt/summaries'
    overfit: bool = False
    overfit_num_samples: int = 12
    load_single_camera: bool = False
    features: DatasetFeatures = DatasetFeatures(
        is_itw = False, 
        has_dhhc_sig = True,
        has_op_kpts = True,
    )

@dataclass
class Demo:
    original_data_folder: str  = ''
    number_of_regions: int = 75
    image_folder: str = 'images'
    bev_folder: str = 'bev'
    openpose_folder: str = 'keypoints/keypoints'
    vitpose_folder: str = 'vitpose'
    image_format: str = 'png'
    image_name_select: str = ''
    has_gt_contact_annotation: bool = False
    imar_vision_datasets_tools_folder: str = 'essentials/imar_vision_datasets_tools' #'datasets/original'
    unique_keypoint_match: bool = True
