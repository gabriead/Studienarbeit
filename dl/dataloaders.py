from typing import List, Tuple, Union
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl


# PyTorch Dataset class for time series data


class TimeSeriesDataset:
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        window: int,
        horizon: int,
        n_val: Union[float, int] = 0.2,
        n_test: Union[float, int] = 0.2,
        normalize: str = "None",  # options are "none", "local", "global"
        normalize_params: Tuple[
            float, float
        ] = None,  # tuple of mean and std for pre-calculated standardization
        mode="train",  # options are "train", "val", "test"
    ):
        if isinstance(n_val, float):
            n_val = int(n_val * len(data))
        if isinstance(n_test, float):
            n_test = int(n_test * len(data))
        if isinstance(data, pd.DataFrame):
            data = data.values
        if data.ndim==1:
            data = data.reshape(-1,1)
        if normalize == "global" and mode != "train":
            assert (
                isinstance(normalize_params, tuple)
                and len(normalize_params) == 2
            ), "If using Global Normalization, in valid and test mode normalize_params argument should be a tuple of precalculated mean and std"
        self.data = data.copy()
        self.n_val = n_val
        self.n_test = n_test
        self.window = window
        self.horizon = horizon
        self.normalize = normalize
        self.mode = mode
        total_data_set_length = len(data)
        # The beginning of the data set is where 'train' starts
        # The end of the dataset is here we find the last testing data
        # We therefore start at 0
        # And end at total_data_set_length = n_samples + (n_model+1) + n_val + n_test
        # (a sample is n_model vectors for X and 1 vector for Y)
        # Final -1 is to reflect Python's 0-array convention
        self.n_samples = (
            total_data_set_length - (self.horizon + 1) - self.n_val - self.n_test
        )
        # Adjust the start of the dataset for training / val / test
        if mode == "train":
            start_index = 0
            end_index = (self.horizon + 1) + self.n_samples

        elif mode == "val":
            start_index = (self.horizon + 1) + self.n_samples - self.window
            end_index = (self.horizon + 1) + self.n_samples + self.n_val

        elif mode == "test":
            start_index = (self.horizon + 1) + self.n_samples + self.n_val - self.window
            end_index = (self.horizon + 1) + self.n_samples + self.n_val + self.n_test

        # This is the actual input on which to iterate
        self.data = data[start_index:end_index, :]
        if normalize == "global":
            if mode == "train":
                self.mean = data.mean()
                self.std = data.std()
            else:
                self.mean, self.std = normalize_params
            self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.horizon - self.window + 1 #to account for zero indexing

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window, :]
        y = None
        y = self.data[idx + self.window : idx + self.window + self.horizon, :]
        if self.normalize == "local":
            x = (x - x.mean()) / x.std()
            y = (y - y.mean()) / y.std()
        return x, y


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        n_val: Union[float, int] = 0.2,
        n_test: Union[float, int] = 0.2,
        window: int = 10,
        horizon: int = 1,
        normalize: str = "none",
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data = data
        self.n_val = n_val
        self.n_test = n_test
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self._is_global = normalize=="global"

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = TimeSeriesDataset(
                data=self.data,
                window=self.window,
                horizon=self.horizon,
                n_val=self.n_val,
                n_test=self.n_test,
                normalize=self.normalize,
                normalize_params= None,
                mode="train",
            )
            self.val = TimeSeriesDataset(
                data=self.data,
                window=self.window,
                horizon=self.horizon,
                n_val=self.n_val,
                n_test=self.n_test,
                normalize=self.normalize,
                normalize_params= (self.train.mean, self.train.std) if self._is_global else None,
                mode="val",
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = TimeSeriesDataset(
                data=self.data,
                window=self.window,
                horizon=self.horizon,
                n_val=self.n_val,
                n_test=self.n_test,
                normalize=self.normalize,
                normalize_params= (self.train.mean, self.train.std) if self._is_global else None,
                mode="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
