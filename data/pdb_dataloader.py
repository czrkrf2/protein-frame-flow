"""PDB data loader."""
import math
import torch
import tree
import numpy as np
import torch
import pandas as pd
import logging

from data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist


class PdbDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        """
        Initialize the PdbDataModule.

        Parameters:
            data_cfg: Configuration object containing data parameters.
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset
        self.sampler_cfg = data_cfg.sampler

    def setup(self, stage: str):
        """
        Set up the datasets for training and validation.

        Parameters:
            stage (str): Stage of the model lifecycle ('fit', 'validate', etc.).
        """
        self._train_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=True,
        )
        self._valid_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=False,
        )

    def train_dataloader(self, rank=None, num_replicas=None):
        """
        Create the training data loader.

        Parameters:
            rank (int, optional): Rank of the current process.
            num_replicas (int, optional): Total number of processes.

        Returns:
            DataLoader: Training data loader.
        """
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._train_dataset,
            batch_sampler=LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
                rank=rank,
                num_replicas=num_replicas,
            ),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self):
        """
        Create the validation data loader.

        Returns:
            DataLoader: Validation data loader.
        """
        return DataLoader(
            self._valid_dataset,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )


class PdbDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
        ):
        """
        Initialize the PdbDataset.

        Parameters:
            dataset_cfg: Configuration object for the dataset.
            is_training (bool): Whether this is a training dataset.
        """
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self._init_metadata()
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        """
        Check if this is a training dataset.

        Returns:
            bool: True if training dataset, False otherwise.
        """
        return self._is_training

    @property
    def dataset_cfg(self):
        """
        Get the dataset configuration.

        Returns:
            object: Dataset configuration object.
        """
        return self._dataset_cfg

    def _init_metadata(self):
        """
        Initialize dataset metadata, filtering and sorting the data.
        """
        # Process CSV with different filtering criterions.
        pdb_csv = pd.read_csv(self.dataset_cfg.csv_path)
        self.raw_csv = pdb_csv
        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.dataset_cfg.max_num_res]
        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= self.dataset_cfg.min_num_res]
        if self.dataset_cfg.subset is not None:
            pdb_csv = pdb_csv.iloc[:self.dataset_cfg.subset]
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)

        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv
            self._log.info(
                f'Training: {len(self.csv)} examples')
        else:
            eval_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.dataset_cfg.min_eval_length]
            all_lengths = np.sort(eval_csv.modeled_seq_len.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = eval_csv[eval_csv.modeled_seq_len.isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self.dataset_cfg.samples_per_eval_length, replace=True, random_state=123)
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')

    def _process_csv_row(self, processed_file_path):
        """
        Process a single CSV row to extract and transform features.

        Parameters:
            processed_file_path (str): Path to the processed file.

        Returns:
            dict: Processed features.
        """
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats)

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[min_idx:(max_idx+1)], processed_feats)

        # Run through OpenFold data transforms.
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        res_idx = processed_feats['residue_index']
        return {
            'aatype': chain_feats['aatype'],
            'res_idx': res_idx - np.min(res_idx) + 1,
            'rotmats_1': rotmats_1,
            'trans_1': trans_1,
            'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
        }

    def __len__(self):
        """
        Get the number of examples in the dataset.

        Returns:
            int: Number of examples.
        """
        return len(self.csv)

    def __getitem__(self, idx):
        """
        Get an example by index.

        Parameters:
            idx (int): Index of the example.

        Returns:
            dict: Example features.
        """
        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row['processed_path']
        chain_feats = self._process_csv_row(processed_file_path)
        chain_feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx
        return chain_feats


class LengthBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        """
        Initialize the LengthBatcher.

        Parameters:
            sampler_cfg: Configuration object for the sampler.
            metadata_csv: CSV containing metadata.
            seed (int): Random seed.
            shuffle (bool): Whether to shuffle batches.
            num_replicas (int, optional): Total number of replicas.
            rank (int, optional): Rank of the current process.
        """
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        self._num_batches = math.ceil(len(self._data_csv) / self.num_replicas)
        self._data_csv['index'] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')
        
    def _replica_epoch_batches(self):
        """
        Create batches for the current replica for the current epoch.

        Returns:
            list: List of batches, each batch is a list of indices.
        """
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self._data_csv), generator=rng).tolist()
        else:
            indices = list(range(len(self._data_csv)))

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[
                indices[self.rank::self.num_replicas]
            ]
        else:
            replica_csv = self._data_csv
        
        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist()
                sample_order.append(batch_indices)
        
        # Remove any length bias.
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        return [sample_order[i] for i in new_order]

    def _create_batches(self):
        """
        Create all batches for the current epoch.
        """
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        """
        Iterate over the batches for the current epoch.

        Returns:
            iterator: Iterator over batches.
        """
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        """
        Get the number of batches.

        Returns:
            int: Number of batches.
        """
        return len(self.sample_order)
