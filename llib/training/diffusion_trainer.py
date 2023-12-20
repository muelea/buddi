import time
import torch
import os.path as osp
from tqdm import tqdm 
from loguru import logger as guru
from torch.utils.data import DataLoader
import torch.nn as nn

from llib.optimizer.build import build_optimizer
from llib.models.build import build_model
from llib.data.collective import PartitionSampler

class Trainer(nn.Module):
    def __init__(
        self,
        train_cfg,
        train_module,
        optimizer,
        logger,
        device,
        batch_size,
    ):
        super().__init__()
        """
        Takes SMPL parameters as input and outputs SMPL parameters as output.
        Parameters
        ----------
        train_cfg : DictConfig
            Training configuration.
        train_module : nn.Module that implements a single training step. A model
            a loss function, and the datasets should be member variables.
        """

        self.train_cfg = train_cfg
        self.device = device
        self.batch_size = batch_size

        # training
        self.train_module = train_module
        self.optimizer = optimizer
        self.logger = logger

        # current model and optimizers
        # logger uses these dicts to save and load checkpoints
        self.optimizers_dict = {'optimizer': self.optimizer}

        # training params
        self.endtime = time.time() + train_cfg.max_duration
        self.max_epochs = train_cfg.max_epochs

        # when output folder is not emply / has a checkpoint resume=True
        self.pretrained = train_cfg.pretrained
        latest_checkpoint = self.logger.get_latest_checkpoint()
        self.resume = True if latest_checkpoint is not None else False

        # training state
        self.epoch = 0
        self.batch_idx = 0
        self.steps = 0
        self.checkpoint = None

        # Store histrogram data for tensorboard
        self.histograms = {}

        # Finally, configure training (load checkpoints, etc.)
        self.configure_training()

    def append_histrogram_data(self, histogram_data={}, add_model_weights=True):
        if len(self.histograms) == 0: # steps doen't work when training is resumed
            # setup storage for parameters that we monitor with histrograms
            for name, val in histogram_data.items():
                self.histograms[name] = [val]
            # setup storage for model parameters
            if add_model_weights:
                for name, weight in self.train_module.named_parameters():
                    if weight.grad is not None:
                        object_name, param_name = name.split('.')[0], '.'.join(name.split('.')[1:])
                        self.histograms[f'{object_name}/{param_name}'] = [weight]
                        self.histograms[f'{object_name}/{param_name}.grad'] = [weight.grad]
        else:
            if add_model_weights:
                for name, weight in self.train_module.named_parameters():
                    if weight.grad is not None: 
                        object_name, param_name = name.split('.')[0], '.'.join(name.split('.')[1:])
                        self.histograms[f'{object_name}/{param_name}'].append(weight)
                        self.histograms[f'{object_name}/{param_name}.grad'].append(weight.grad)
            for name, val in histogram_data.items():
                self.histograms[name].append(val)

    def configure_training(self):
        """
        If checkpoint folder contains a checkpoint, resume training from it.
        If a checkpoint path is provided, load the checkpoint and resume training.
        If a pretrained model path is provided, load the pretrained model.
        If neither of the above is true, start training from scratch.
        """

        if self.resume:
            if self.pretrained == '':
                latest_checkpoint_path = self.logger.get_latest_checkpoint()
                self.resume_training(checkpoint_path=latest_checkpoint_path)
                guru.info(f'Resume training from latest checkpoint {latest_checkpoint_path}.')
            else:
                self.resume_training(checkpoint_path=self.pretrained)
                guru.info(f'Resume training from pretrained checkpoint {self.pretrained}.')
        else:
            if self.pretrained == '':
                guru.info('Starting training from scratch.')
            else:
                self.load_pretrained_model_weights(checkpoint_path=self.pretrained)
                guru.info(f'Load model weights from pretrained checkpoint {self.pretrained}.')


    def load_pretrained_model_weights(self, checkpoint_path):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        self.checkpoint = self.logger.load_checkpoint_weights(
            train_module=self.train_module,
            checkpoint_file=checkpoint_path
        )

    def resume_training(self, checkpoint_path):
        """
        Resume training from a checkpoint. If no checkpoint is provided,
        resume from the latest checkpoint in the checkpoint directory.
        """

        self.checkpoint = self.logger.load_checkpoint(
            train_module=self.train_module,
            optimizers=self.optimizers_dict, 
            checkpoint_file=checkpoint_path)
        self.steps = self.checkpoint['total_steps']

        steps_per_epoch = int(len(self.train_module.train_ds) // self.batch_size)
        if self.checkpoint['batch_idx'] == steps_per_epoch:
            self.epoch = self.checkpoint['epoch'] + 1
            self.batch_idx = 0
        else:
            self.epoch = self.checkpoint['epoch']
            self.batch_idx = self.checkpoint['batch_idx']

    def train(self):
        """Full training process."""

        # get validation and summary steps from frequency
        self.steps_per_epoch = int(len(self.train_module.train_ds) // self.batch_size)
        self.summary_steps = int(self.logger.sum_freq * self.steps_per_epoch)
        self.checkpoint_steps = int(self.logger.ckpt_freq * self.steps_per_epoch)
        guru.info(f'Saving summaries every {self.summary_steps} steps.')
        guru.info(f'Saving checkpoints every {self.checkpoint_steps} steps.')
        guru.info(f'One epoch has {self.steps_per_epoch} steps.')

        # check if freqs and shuffle are compatible
        if self.train_cfg.shuffle_train:
            assert self.logger.sum_freq >= 1.0 or self.logger.ckpt_freq >= 1.0, \
                'Does not support training data shuffling when logging frequency < 1.0 and > 0.0'

        # Run training for num_epochs epochs
        epochs = range(self.epoch, self.max_epochs)
        for epoch in epochs: #tqdm(epochs, total=self.max_epochs, initial=self.epoch):
            guru.info(f'================== TRAIN EPOCH {epoch}/{self.max_epochs} ({self.steps} steps) =====================')
            self.epoch = epoch
            self.train_one_epoch()

    def dict_to_device(self, batch):
        """Move batch to device."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                batch[k] = self.dict_to_device(v)
            #else isinstance(v, list):
            #    batch[k] = [x.to(self.device) for x in v]
            else:
                batch[k] = v
        return batch

    def expand_batch(self, batch, padding):
        """Expand batch elements by padding value."""
        # for each element in batch add zeros along axis 0 to match pad size
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = torch.cat([v, torch.zeros((padding, *v.shape[1:]), device=self.device)], dim=0)
            elif isinstance(v, dict):
                batch[k] = self.expand_batch(v, padding)
            elif isinstance(v, list):
                batch[k] = v + [v[0]] * padding
            else:
                print(k, v)
                #raise NotImplementedError

        return batch


    @torch.no_grad()
    def validate(self, is_training=True):
        """Validate all datasets."""

        # set model to evaluation mode
        self.train_module.eval()
        drop_last = True if is_training else False

        ckpt_metric = 0.0
        if self.train_module.val_ds is not None:
            for val_ds_name, val_ds in self.train_module.val_ds.items():

                self.train_module.evaluator.reset()

                # check if dataset is large enough for batch size
                expand_batch = False
                if len(val_ds) < self.batch_size:
                    guru.warning(f'Validation dataset {val_ds_name} is smaller than batch size {self.batch_size}. Expanding batch.')
                    expand_batch = True
                    drop_last = False

                # load validation datasets
                val_loader = DataLoader(
                    val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.train_cfg.num_workers,
                    pin_memory=self.train_cfg.pin_memory,
                    drop_last=drop_last
                )

                # validate
                for batch_idx, batch in enumerate(val_loader):
                    

                    if batch_idx == 0:
                        batch = self.dict_to_device(batch)
                        batch = self.expand_batch(batch, self.batch_size - len(val_ds)) if expand_batch else batch
                        self.train_module.single_validation_step(batch)
                
                self.train_module.evaluator.final_accumulate_step()

                if is_training:
                    # get main validation metric
                    #ckpt_metric_name = self.train_module.evaluator.cfg.checkpoint_metric
                    #ckpt_metric = torch.tensor(self.train_module.evaluator.accumulator[ckpt_metric_name]).mean()
                    ckpt_metric = self.train_module.evaluator.ckpt_metric_value
                    # self.logger.tsw.add_scalar(f'val/ckpt_metric', ckpt_metric, self.steps)
                    self.logger.log('scalar', f'val_setting/{val_ds_name}_lr', self.optimizer.param_groups[0]['lr'], self.steps)
                    self.logger.log('scalar', f'val_setting/{val_ds_name}_ckpt_metric', ckpt_metric, self.steps)
                    self.logger.log('scalar', f'val_setting/{val_ds_name}_epoch', self.epoch, self.steps)

                    # add metric to tensorboard
                    for k, v in self.train_module.evaluator.accumulator.items():
                        tbk = f'loss/{k}' if 'loss' in k else f'metric/{k}'
                        # self.logger.tsw.add_scalar(f'val/{tbk}', v, self.steps)
                        self.logger.log('scalar', f'val/{val_ds_name}_{tbk}', v, self.steps)

                    val_tb_output = self.train_module.evaluator.tb_output
                    if val_tb_output is not None:
                        # render_per_ds = {f'{val_ds_name}_{k}': v for k, v in val_tb_output['images'].items()}
                        self.add_summary_images(
                            val_tb_output['images'], 
                            split='val', 
                            max_images=min(12, self.batch_size), 
                            ds_name=val_ds_name
                        )
                        # self.add_summary_images(render_per_ds, split='val', max_images=min(12, self.batch_size))

                    # self.train_module.train()
                    # return ckpt_metric

                else:
                    # add print results on validation set
                    for k, v in self.train_module.evaluator.accumulator.items():
                        guru.info(f'Validation Set Error: {k} = {v.mean():.4f}')

        if is_training:            
            self.train_module.train()
            return ckpt_metric



    def train_one_epoch(self):
        """Single epoch training step."""

        # model to training mode
        self.train_module.train()

        # create / reset sampler
        sampler = PartitionSampler(
            ds_names=self.train_module.train_ds.dataset_list, 
            ds_lengths=self.train_module.train_ds.ds_lengths,
            ds_partition=self.train_module.train_ds.orig_partition,
            shuffle=self.train_cfg.shuffle_train,
            batch_size=self.batch_size
        )

        # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
        data_loader = DataLoader(
            self.train_module.train_ds,
            batch_size=self.batch_size,
            shuffle=False, # suffling is done in data class
            num_workers=self.train_cfg.num_workers,
            pin_memory=self.train_cfg.pin_memory,
            drop_last=False, # was false for bs 64
            sampler=sampler
        )
        
        # skip first batches if batch_idx is not 0
        if self.batch_idx > 0:
            for _ in range(self.batch_idx):
                next(iter(data_loader))

        # Iterate over all batches in an epoch
        for batch_idx, batch in enumerate(data_loader, start=self.batch_idx):    

            # move input to device
            batch = self.dict_to_device(batch)

            # make the training step: regressor - loop
            loss, loss_dict, output = self.train_module.single_training_step(batch)

            # backprop regressor
            self.optimizer.zero_grad()
            loss.backward()

            # clip gradients
            if self.train_cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.train_module.parameters(), 
                    self.train_cfg.clip_grad_norm,
                    error_if_nonfinite=True
                )

            self.optimizer.step()
            self.steps += 1

            # print and log loss values  
            #self.logger.print_loss_dict(loss_dict, self.epoch, self.steps)
            for loss_name, loss_value in loss_dict.items():
                # self.logger.tsw.add_scalar(f'train/{loss_name}', loss_value, self.steps)
                self.logger.log('scalar', f'train/{loss_name}', loss_value, self.steps)

            # save training summaries
            # self.logger.tsw.add_scalar(f'train_setting/lr', self.optimizer.param_groups[0]['lr'], self.steps)
            self.logger.log('scalar', f'train_setting/lr', self.optimizer.param_groups[0]['lr'], self.steps)
            self.logger.log('scalar', f'train_setting/epoch', self.epoch, self.steps)
            #if 'histograms' in output.keys():
            #    self.append_histrogram_data(output['histograms'], add_model_weights=True)

            # Tensorboard logging every summary_steps steps
            if self.summary_steps > 0 and self.steps % self.summary_steps == 0:

                guru.info(f'Add train summary ({self.epoch}/{self.max_epochs} epochs; {self.steps} steps) ...')

                if 'images' in output.keys():
                    self.add_summary_images(output['images'], split='train', max_images=min(12, self.batch_size))
                for name, values in self.histograms.items(): # add histograms
                    if len(values) > 0:
                        # self.logger.tsw.add_histogram(name, torch.cat(values, dim=0), self.steps)
                        self.logger.log('histogram', name, torch.cat(values, dim=0), self.steps)
                        self.histograms[name] = []

            # validate and save checkpoint
            if self.checkpoint_steps > 0 and self.steps % self.checkpoint_steps == 0:

                guru.info(f'Run validation ({self.epoch}/{self.max_epochs} epochs; {self.steps} steps) ...')

                ckpt_metric = self.validate()
                # val_output = self.train_module.evaluator.tb_output

                # save validation images
                # if val_output is not None:
                    # self.add_summary_images(val_output['images'], split='val', max_images=min(12, self.batch_size))

                # save checkpoint
                self.logger.save_checkpoint(self.train_module, self.optimizers_dict,
                    self.epoch, batch_idx+1, self.batch_size, self.steps, ckpt_metric)

    def add_summary_images(self, output, split='train', max_images=32, ds_name=''):
        """Write summary to Tensorboard."""
        # if isinstance(output, dict):
        #     for k, v in output.items():
        #         images = self.train_module.render_output(v, max_images=max_images)
        #         for img_name, img in images.items():
        #             if self.logger.logger_type == 'tensorboard':
        #                 img = torch.from_numpy(img).unsqueeze(0)[..., :3] / 255
        #             else:
        #                 img = img[:,:,:3]
        #             self.logger.log(
        #                 'images', f'{split}/{k}_{img_name}', img, self.steps, dataformats='NHWC'
        #             )
        #             self.logger.log(
        #                 'scalar', f'{split}/{k}_{img_name}_epoch', self.epoch, self.steps
        #             )
        # else:
        images = self.train_module.render_output(output, max_images=max_images)
        for img_name, img in images.items():
            if self.logger.logger_type == 'tensorboard':
                img = torch.from_numpy(img).unsqueeze(0)[..., :3] / 255
            else:
                img = img[:,:,:3]
            self.logger.log(
                'images', f'{split}/{ds_name}{img_name}', img, self.steps, dataformats='NHWC'
            )
            #self.logger.log(
            #    'scalar', f'{split}/{ds_name}{img_name}_epoch', self.epoch, self.steps
            #)