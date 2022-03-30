"""UCI datasets as used in Bayesian regression

Code adapted from https://github.com/yaringal/DropoutUncertaintyExps under CC BY-NC 4.0
license.
"""
import pathlib

import numpy as np
import torch

DATASETS = {
    "housing": "bostonHousing",
    "carbon": "carbon",
    "concrete": "concrete",
    "energy": "energy",
    "kin8m": "kin8nm",
    "naval": "naval",
    "power": "power-plant",
    "protein": "protein-tertiary-structure",
    "superconductivity": "superconductivity",
    "wine-red": "wine-quality-red",
    "wine-white": "wine-white",
    "yacht": "yacht",
}

N_SPLITS = {
    "housing": 20,
    "carbon": 20,
    "concrete": 20,
    "energy": 20,
    "kin8m": 20,
    "naval": 20,
    "power": 20,
    "protein": 5,
    "superconductivity": 20,
    "wine-red": 20,
    "wine-white": 20,
    "yacht": 20,
}

DEFAULT_DATA_DIR = pathlib.Path("./data")


class UCICollection:
    def __init__(
        self,
        name,
        test_split_idx=0,
        train_val_split=0.8,
        standardize_inputs=True,
        standardize_targets=True,
        data_dir=None,
    ):
        if name not in DATASETS:
            raise ValueError(f"Unknown UCI dataset {name}")

        self.name = name

        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        else:
            data_dir = pathlib.Path(data_dir)
        self.data_folder = data_dir / "UCI_Datasets" / DATASETS[name] / "data"

        X, y = self._load_dataset(self.data_folder)

        n_splits = int(np.loadtxt(self.data_folder / "n_splits.txt"))
        if test_split_idx >= n_splits:
            raise ValueError(f"Dataset has only {n_splits} splits defined")

        X_train, y_train, X_test, y_test = self._load_predefined_split(
            X, y, self.data_folder, test_split_idx
        )
        self.input_dim = X_train.shape[1]
        self.target_dim = y_train.shape[1]
        self.n_training_samples = int(train_val_split * X_train.shape[0])

        if standardize_inputs:
            mean = X_train[: self.n_training_samples].mean(axis=0)
            std = X_train[: self.n_training_samples].std(axis=0)
            std[std == 0] = 1
            self.X_train = (X_train - mean) / std
            self.X_test = (X_test - mean) / std
        else:
            self.X_train = X_train
            self.X_test = X_test

        if standardize_targets:
            self.target_mean = y_train[: self.n_training_samples].mean(axis=0)
            self.target_std = y_train[: self.n_training_samples].std(axis=0)
            y_train[: self.n_training_samples] = (
                y_train[: self.n_training_samples] - self.target_mean
            ) / self.target_std
            self.y_train = y_train
        else:
            self.y_train = y_train
            self.target_mean = None
            self.target_std = None
        self.y_test = y_test

    @property
    def train_data(self):
        dataset = self._init_tensor_dataset(
            self.X_train[: self.n_training_samples],
            self.y_train[: self.n_training_samples],
        )
        if self.target_mean is not None:
            dataset.target_mean = self.target_mean
            dataset.target_std = self.target_std

        return dataset

    @property
    def eval_data(self):
        if self.n_training_samples < len(self.X_train):
            return self._init_tensor_dataset(
                self.X_train[self.n_training_samples :],
                self.y_train[self.n_training_samples :],
            )
        else:
            return None

    @property
    def test_data(self):
        return self._init_tensor_dataset(self.X_test, self.y_test)

    def _load_dataset(self, path):
        data = np.loadtxt(path / "data.txt")
        index_features = np.loadtxt(path / "index_features.txt")
        index_target = np.loadtxt(path / "index_target.txt")

        X = data[:, [int(i) for i in index_features.tolist()]]

        index_target = index_target.tolist()
        if isinstance(index_target, float):
            index_target = [index_target]
        y = data[:, [int(i) for i in index_target]]

        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        return X, y

    def _load_predefined_split(self, X, y, path, split_idx):
        index_train = np.loadtxt(path / f"index_train_{split_idx}.txt")
        X_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]]

        index_test = np.loadtxt(path / f"index_test_{split_idx}.txt")
        X_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]]

        return X_train, y_train, X_test, y_test

    def _init_tensor_dataset(self, x, y):
        inps = torch.from_numpy(x).float()
        tgts = torch.from_numpy(y).float()

        return torch.utils.data.TensorDataset(inps, tgts)
