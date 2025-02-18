import random
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torch_geometric.loader import DataLoader

from liflow.data.dataset import NoisedFrameDataset, TimeDelayedPairDataset
from liflow.utils.prior import (
    AdaptiveMaxwellBoltzmannPrior,
    MaxwellBoltzmannPrior,
    Prior,
    UniformScaleNormalPrior,
)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() & 0xFFFFFFFF
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.load_data()  # reload npz arrays


class NoisedFrameDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        index_files: list[str],
        time_delay_steps: int = 100,
        cutoff: float = 5.0,
        pbc: bool = True,
        in_memory: bool = False,
        batch_size: int = 16,
        num_train_samples: int = 4000,
        num_valid_samples: int = 400,
        train_valid_split: bool = True,
        sample_weight_comp: bool = False,
        num_workers: int = 0,
        seed: int = 42,
        noise: UniformScaleNormalPrior = UniformScaleNormalPrior(),
        prior: Prior = MaxwellBoltzmannPrior(scale=0.1),
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.index_files = index_files
        self.time_delay_steps = time_delay_steps
        self.cutoff = cutoff
        self.pbc = pbc
        self.in_memory = in_memory
        self.batch_size = batch_size
        self.num_train_samples = num_train_samples
        self.num_valid_samples = num_valid_samples
        self.train_valid_split = train_valid_split
        self.sample_weight_comp = sample_weight_comp
        self.num_workers = num_workers
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.noise = noise
        self.prior = prior

    def setup(self, stage: str = None):
        # Load train + valid data
        df = pd.concat([pd.read_csv(self.data_path / f) for f in self.index_files])
        if self.train_valid_split:
            # select num_valid_samples from df_train
            self.df_valid = df.sample(
                n=self.num_valid_samples, replace=False, random_state=self.seed
            ).copy()
            self.df_train = df.drop(self.df_valid.index).copy()
        else:
            self.df_train = df.copy()

        # Sample weights for training
        self.weights = np.ones(len(self.df_train), dtype=float)
        if self.sample_weight_comp:
            comp_count = self.df_train.groupby("comp").transform("count")["name"]
            self.weights *= 1.0 / comp_count.to_numpy()

        dataset_kwargs = dict(
            data_path=self.data_path,
            time_delay_steps=self.time_delay_steps,
            noise=self.noise,
            prior=self.prior,
            cutoff=self.cutoff,
            pbc=self.pbc,
            in_memory=self.in_memory,
        )

        if stage in ["fit", "validate"]:
            self.train_dataset = NoisedFrameDataset(df=self.df_train, **dataset_kwargs)
            if self.train_valid_split:
                self.valid_dataset = NoisedFrameDataset(
                    df=self.df_valid, **dataset_kwargs
                )
            else:
                self.valid_dataset = self.train_dataset
        else:  # test
            raise NotImplementedError

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.weights,
            num_samples=self.num_train_samples,
            replacement=True,
            generator=self.generator,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        if self.train_valid_split:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
            )
        else:  # sample from train dataset
            sampler = RandomSampler(
                self.valid_dataset,
                replacement=True,
                num_samples=self.num_valid_samples,
                generator=self.generator,
            )
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker,
                generator=self.generator,
                sampler=sampler,
                pin_memory=True,
            )

    def test_dataloader(self):
        raise NotImplementedError


class TimeDelayedPairDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        index_files: list[str],
        time_delay_steps: int = 100,
        cutoff: float = 5.0,
        pbc: bool = True,
        in_memory: bool = False,
        batch_size: int = 16,
        num_train_samples: int = 4000,
        num_valid_samples: int = 400,
        train_valid_split: bool = True,
        sample_weight_comp: bool = False,
        num_workers: int = 0,
        seed: int = 42,
        prior: Prior = AdaptiveMaxwellBoltzmannPrior(),
        neighbor_list_both_ends: bool = False,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.index_files = index_files
        self.time_delay_steps = time_delay_steps
        self.cutoff = cutoff
        self.pbc = pbc
        self.in_memory = in_memory
        self.batch_size = batch_size
        self.num_train_samples = num_train_samples
        self.num_valid_samples = num_valid_samples
        self.train_valid_split = train_valid_split
        self.sample_weight_comp = sample_weight_comp
        self.num_workers = num_workers
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.prior = prior
        self.neighbor_list_both_ends = neighbor_list_both_ends

    def setup(self, stage: str = None):
        # Load train + valid data
        df = pd.concat([pd.read_csv(self.data_path / f) for f in self.index_files])
        if self.train_valid_split:
            # select num_valid_samples from df_train
            self.df_valid = df.sample(
                n=self.num_valid_samples, replace=False, random_state=self.seed
            ).copy()
            self.df_train = df.drop(self.df_valid.index).copy()
        else:
            self.df_train = df.copy()

        # Sample weights for training
        self.weights = np.ones(len(self.df_train), dtype=float)
        if self.sample_weight_comp:
            comp_count = self.df_train.groupby("comp").transform("count")["name"]
            self.weights *= 1.0 / comp_count.to_numpy()

        dataset_kwargs = dict(
            data_path=self.data_path,
            time_delay_steps=self.time_delay_steps,
            prior=self.prior,
            cutoff=self.cutoff,
            pbc=self.pbc,
            neighbor_list_both_ends=self.neighbor_list_both_ends,
            in_memory=self.in_memory,
        )

        if stage in ["fit", "validate"]:
            self.train_dataset = TimeDelayedPairDataset(
                df=self.df_train, **dataset_kwargs
            )
            if self.train_valid_split:
                self.valid_dataset = TimeDelayedPairDataset(
                    df=self.df_valid, **dataset_kwargs
                )
            else:
                self.valid_dataset = self.train_dataset
        else:  # test
            raise NotImplementedError

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.weights,
            num_samples=self.num_train_samples,
            replacement=True,
            generator=self.generator,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        if self.train_valid_split:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
            )
        else:  # sample from train dataset
            sampler = RandomSampler(
                self.valid_dataset,
                replacement=True,
                num_samples=self.num_valid_samples,
                generator=self.generator,
            )
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker,
                generator=self.generator,
                sampler=sampler,
                pin_memory=True,
            )

    def test_dataloader(self):
        raise NotImplementedError
