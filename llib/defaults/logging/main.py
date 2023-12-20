import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass

@dataclass 
class Logging:
    
    # base and run output folder
    # will save output to base_folder
    base_folder: str = 'outdebug'
    run: str = 'run1'

    # subfolders that will be created in base folder
    # folder_freq defines frequency in freq*epochs that results are written.
    # E.g. validation_freq=0.5, runs 2 validations per epoch, 
    # validation_freq=0.0 does not run validations and 
    # validation_freq=1.0 evaluates one time per epoch
    images_folder: str = 'images'
    summaries_folder: str = 'summaries'
    checkpoint_folder: str = 'checkpoints'
    validation_folder: str = 'validation'
    result_folder: str = 'results'

    # frequency in epochs
    # if freq == 0.0, no summary/checkpoints are saved 
    # if freq < 1.0, training data can not be shuffled
    # in this case, set training.shuffle_train = False
    summaries_freq: float = 100.0 # in epochs
    checkpoint_freq: float = 100.0 # in epochs
    #images_freq: float = 0.5 # save images every summary_freq
    #validation_freq: float = 0.5 # validate every checkpoint_freq

    # logger level information
    logger_level: str = 'INFO'

    # which logger to use 
    logger: str = 'tensorboard' # tensorboard, wandb
    project_name: str = 'MyAwesomeProject' # project name for wandb
    run_id: str = 'default' # run id for wandb
    wandb_api_key_path: str = '.wandb/api.txt' # the path to a txt file with your wandb api key

conf = OmegaConf.structured(Logging)