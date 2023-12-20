import os
import yaml
import os.path as osp
import datetime
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import wandb

class Logger():
    def __init__(
        self, 
        cfg,
    ):
        """
        Class that handles saving and loading checkpoints during training.
        Parameters
        ----------
        cfg: cfg
            config file for logger
        """
        self.cfg = cfg
        self.logger_type = cfg.logging.logger

        # basic output folder 
        self.out_folder = osp.join(cfg.logging.base_folder, cfg.logging.run)
        os.makedirs(self.out_folder, exist_ok=True)

        # subfolders
        self.img_folder = osp.join(self.out_folder, cfg.logging.images_folder)
        self.sum_folder = osp.join(self.out_folder, cfg.logging.summaries_folder)
        self.ckpt_folder = osp.join(self.out_folder, cfg.logging.checkpoint_folder)
        self.val_folder = osp.join(self.out_folder, cfg.logging.validation_folder)
        self.res_folder = osp.join(self.out_folder, cfg.logging.result_folder)
        self.create_output_folders()
        
        # frequencies to save images+summaries, checkpoints+validation
        self.sum_freq = cfg.logging.summaries_freq
        self.ckpt_freq = cfg.logging.checkpoint_freq

        # check if checkpoint exists
        self.latest_checkpoint = self.get_latest_checkpoint()

        # setup logging helpers
        if self.logger_type == 'tensorboard':
            self.tsw = SummaryWriter(log_dir=self.sum_folder) # use tensorboard to monitor training
        elif self.logger_type == 'wandb':
            os.environ["WANDB_SILENT"] = "true"
            # with open('.wandb/api.txt') as f:
            with open(cfg.logging.wandb_api_key_path) as f:
                key = f.readlines()
                key = key[0].strip('\n')
            wandb.login(key=key)

            # check if wandb project exists
            # get run_id if it exists or create a new one
            if self.exists_checkpoint():
                # read old config and get run_id
                with open(osp.join(self.out_folder, 'config.yaml')) as f:
                    old_cfg = OmegaConf.load(f)
                run_id = old_cfg.logging.run_id
                logger.info(f'Resume training logs in wandb with run_id: {run_id}')
            else:
                run_id = wandb.util.generate_id()
                cfg.logging.run_id = run_id
                logger.info(f'Create new wandb project with run_id: {run_id}')

            # init wandb
            self.wdb = wandb.init(
                project=cfg.logging.project_name, 
                config=OmegaConf.to_container(cfg, resolve=True),
                name=cfg.logging.run,
                id=run_id,
                resume='allow',
            )
        
        # save config after wandb init and after run_id was created
        self.save_config() # write config file to output_folder

    def log(self, dtype, name, value, step=None, dataformats='NHWC'):
        """
        Log a value to tensorboard or wandb.
        Parameters
        ----------
        name: str
            name of value
        value: float
            value to log
        step: int
            step to log value at
        type: str
            type of value to log (scalar, image, histogram, etc.)
        """

        if self.logger_type == 'tensorboard':
            if dtype == 'scalar':
                self.tsw.add_scalar(name, value, step)
            elif dtype == 'histogram':
                self.tsw.add_histogram(name, value, step)
            elif dtype == 'images':
                self.tsw.add_images(name, value, step, dataformats=dataformats)
        elif self.logger_type == 'wandb':
            if dtype == 'scalar':
                self.wdb.log({name: value}, step=step)
            elif dtype == 'histogram':
                self.wdb.log({name: wandb.Histogram(value)}, step=step)
            elif dtype == 'images':
                self.wdb.log({name: [wandb.Image(value)]}, step=step)

    def create_output_folders(self):
        os.makedirs(self.out_folder, exist_ok=True)
        os.makedirs(self.img_folder, exist_ok=True)
        os.makedirs(self.sum_folder, exist_ok=True)
        os.makedirs(self.ckpt_folder, exist_ok=True)
        os.makedirs(self.val_folder, exist_ok=True)
        os.makedirs(self.res_folder, exist_ok=True)

    def save_config(self):
        config_file_path = osp.join(self.out_folder, 'config.yaml')
        if not osp.exists(config_file_path):
            with open(config_file_path, 'w') as f:
                OmegaConf.save(self.cfg, f)
        else:
            new_config_file_path = osp.join(self.out_folder, 'config_latest_run.yaml')
            if not osp.exists(new_config_file_path):
                with open(new_config_file_path, 'w') as f:
                    OmegaConf.save(self.cfg, f)
            logger.warning('Config file already exists. Not overwriting config.yaml. Save settings to config_latest_run.yaml.')
            
    def get_latest_checkpoint(self):
        """Get filename of latest checkpoint if it exists."""
        checkpoints = [] 
        for dirpath, dirnames, filenames in os.walk(self.ckpt_folder):
            for filename in filenames:
                if filename.endswith('.pt'):
                    checkpoints.append(osp.abspath(osp.join(dirpath, filename)))
        checkpoints = sorted(checkpoints)
        latest_checkpoint =  None if (len(checkpoints) == 0) else checkpoints[-1]
        return latest_checkpoint

    def exists_checkpoint(self, checkpoint_file=None):
        """Check if a checkpoint exists in the current directory."""
        if checkpoint_file is None:
            return False if self.latest_checkpoint is None else True
        else:
            return osp.isfile(checkpoint_file)

    def load_checkpoint_weights(self, train_module, checkpoint_file):
        """Load a checkpoint."""

        checkpoint = torch.load(checkpoint_file)
        for model in train_module.trainable_params:
            if model in checkpoint:
                eval(f'train_module.{model}').load_state_dict(checkpoint[model], strict=False)
            else:
                logger.warning(f'Model {model} not found in checkpoint.')
        return {}

    def load_checkpoint(self, train_module, optimizers, checkpoint_file=None):
        """Load a checkpoint."""

        if checkpoint_file is None:
            checkpoint_file = self.latest_checkpoint
            logger.info(f'Loading latest checkpoint: {checkpoint_file}')

        checkpoint = torch.load(checkpoint_file)

        for model in train_module.trainable_params:
            if model in checkpoint:
                eval(f'train_module.{model}').load_state_dict(checkpoint[model])
            else:
                logger.warning(f'Model {model} not found in checkpoint.')
                
        for optimizer in optimizers:
            if optimizer in checkpoint:
                optimizers[optimizer].load_state_dict(checkpoint[optimizer])
            else:
                logger.warning(f'Optimizer {optimizer} not found in checkpoint.')

        return {'epoch': checkpoint['epoch'],
                'batch_idx': checkpoint['batch_idx'],
                'batch_size': checkpoint['batch_size'],
                'total_steps': checkpoint['total_steps']}

    def get_checkpoint_fn(self, epoch, batch_idx, val_error):
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        ckpt_name = f'{timestamp}__{epoch:010}__{batch_idx:010}__{val_error:.02f}.pt'
        ckpt_path = osp.abspath(osp.join(self.ckpt_folder, ckpt_name))
        logger.info(f'Saving checkpoint to {ckpt_path}')
        return ckpt_path

    def save_checkpoint(self, 
        train_module, optimizers, epoch, batch_idx, batch_size, 
        total_steps, val_error
    ):
        """Save checkpoint."""

        checkpoint = {}

        for model in train_module.trainable_params:
            checkpoint[model] = eval(f'train_module.{model}').state_dict()
        for optimizer in optimizers:
            checkpoint[optimizer] = optimizers[optimizer].state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_idx'] = batch_idx
        checkpoint['batch_size'] = batch_size
        checkpoint['total_steps'] = total_steps

        ckpt_path = self.get_checkpoint_fn(epoch, batch_idx, val_error)
        torch.save(checkpoint, ckpt_path)

    def print_loss_dict(self, ld, stage=0, step=0, abbr=True):
        total_loss = ld['total_loss'].item()
        out = f'Stage/step:{stage:2d}/{step:2} || Tl: {total_loss:.4f} || '
        for k, v in ld.items():
            if k != 'total_loss':
                kprint = ''.join([x[0] for x in k.split('_')]) if abbr else k
                if type(v) == torch.Tensor:
                    v = v.item()
                    out += f'{kprint}: {v:.4f} | '
        print(out)
