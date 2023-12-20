import torch
import numpy as np
from itertools import cycle 

from llib.data.single import SingleDataset
from loguru import logger as guru

class PartitionSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        ds_names=[], 
        ds_lengths=[],
        ds_partition=[],
        shuffle=True,
        batch_size=64,
    ):

        self.ds_names = ds_names
        self.ds_lengths = ds_lengths
        self.ds_partition = ds_partition
        
        assert len(self.ds_names) == len(self.ds_lengths) == len(self.ds_partition)
        assert sum(self.ds_partition) == 1.0

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_batches = int(sum(self.ds_lengths) / batch_size)
        self.ds_cumsum = self._cumsum()
        self.ds_absolute = self._partition_to_absolute()

    def _cumsum(self):
        return np.array([0] + np.cumsum(self.ds_lengths).tolist())
    
    def _partition_to_absolute(self):
        """
        Number of elements per dataset in a batch.
        Round first dataset up/ down to match batch size
        """

        ds_absolute = []

        for dset_idx, ds_name in enumerate(self.ds_names):
            ds_absolute.append(
                round(self.ds_partition[dset_idx] * self.batch_size))
        # If the sum of the elements per dataset is not equal to the batch size
        # fill up with or remove items from first dataset
        if sum(ds_absolute) != self.batch_size:
            error = self.batch_size - sum(ds_absolute)
            ds_zero_new = ds_absolute[0] + error
            ds_absolute[0] = ds_zero_new
                                             
        return ds_absolute
    
    def _prepare_batches(self):
        batch_idxs = []

        dset_idxs = {}
        for dset_idx, ds_name in enumerate(self.ds_names):
            ds_indices = np.arange(self.ds_cumsum[dset_idx], self.ds_cumsum[dset_idx+1])
            if self.shuffle:
                ds_indices = np.random.permutation(ds_indices)
            dset_idxs[ds_name] = cycle(ds_indices)

        for _ in range(self.num_batches):
            curr_idxs = []
            for dset_idx, ds_name in enumerate(self.ds_names):
                for _ in range(self.ds_absolute[dset_idx]):
                    curr_idxs.append(next(dset_idxs[ds_name]))

            curr_idxs = np.array(curr_idxs)
            if self.shuffle:
                np.random.shuffle(curr_idxs)
            batch_idxs.append(curr_idxs)

        if self.shuffle:
            np.random.shuffle(batch_idxs)

        batch_idxs = np.concatenate(batch_idxs)

        return batch_idxs

    def __len__(self):
        #if not hasattr(self, '_batch_idxs'):
        #    self._batch_idxs = self._prepare_batches()
        return self.num_batches * sum(self.ds_absolute) #len(self._batch_idxs)

    def __iter__(self):
        self._batch_idxs = self._prepare_batches()
        return iter(self._batch_idxs)

class CollectivDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        datasets_cfg, 
        split,
        body_model_type,
    ):

        self.datasets_cfg = datasets_cfg
        self.body_model_type = body_model_type
        self.split = split

        # get the list of datasets to be used and their composition parameter
        self.dataset_list = eval(f'datasets_cfg.{self.split}_names')

        if len(self.dataset_list) > 0:
            self.dataset_dict = dict(zip(self.dataset_list, np.arange(len(self.dataset_list))))
            self.datasets = [self.create_single_dataset(ds_name) for ds_name in self.dataset_list]
            self.length = max([len(ds) for ds in self.datasets])
            self.ds_lengths = np.array([len(ds) for ds in self.datasets])
            self.total_length = sum(self.ds_lengths)
            self.total_length_cumsum = self.ds_lengths.cumsum()

        if split == 'train':
            self.orig_partition = eval(f'datasets_cfg.{self.split}_composition')
            self.partition = np.array(self.orig_partition).cumsum()

            guru.info(f'Loading {split} data:')
            for idx in range(len(self.partition)):
                x = self.dataset_list[idx]
                prev = 0 if idx == 0 else self.partition[idx-1]
                y = (self.partition[idx] - prev) * 100
                guru.info(f'  --- {x} share per batch: {y:.02f}% ')


    def create_single_dataset(self, dataset_name):

        # get config file of single dataset
        dataset_cfg = eval(f'self.datasets_cfg.{dataset_name}')

        # create dataset
        dataset = SingleDataset(
            dataset_cfg=dataset_cfg, 
            dataset_name=dataset_name, 
            augmentation=self.datasets_cfg.augmentation,
            image_processing=self.datasets_cfg.processing,
            split=self.split,
            body_model_type=self.body_model_type,
        )

        return dataset

    def __getitem__(self, index):

        ds_idx = int(np.where(index < self.total_length_cumsum)[0][0])

        if ds_idx > 0:
            index = index - self.total_length_cumsum[ds_idx-1]

        item_idx = index % len(self.datasets[ds_idx]) if self.split == 'train' \
            else index # for test and validation set all items are used / no repetitions

        return self.datasets[ds_idx][item_idx]

    def __len__(self):
        return self.total_length
