import numpy as np
import torch
from torchvision import datasets, transforms

DEFAULT_DATA_DIR = "/is/rg/al/Projects/prob-models/data/"


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(
        self, name, split="train", flatten=True, train_split=0.8, data_dir=None
    ):
        assert split in ("train", "val", "test")
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR

        load_train = split == "train" or split == "val"
        if name == "mnist":
            dataset = datasets.MNIST(
                data_dir,
                train=load_train,
                download=True,
                transform=transforms.ToTensor(),
            )
        elif name == "fashion-mnist":
            dataset = datasets.FashionMNIST(
                data_dir,
                train=load_train,
                download=True,
                transform=transforms.ToTensor(),
            )
        else:
            raise ValueError("Unknown dataset name {name}")

        self.images = torch.stack([x[0] for x in dataset], axis=0)
        if split == "train" or split == "val":
            train_samples = int(train_split * len(self.images))
            rng = np.random.RandomState(45)
            idxs = rng.permutation(len(self.images))
            if split == "train":
                train_idxs = idxs[:train_samples]
                self.images = self.images[train_idxs]
            else:
                val_idxs = idxs[train_samples:]
                self.images = self.images[val_idxs]

        self._shape = self.images.shape[1:]

        if flatten:
            self.images = self.images.reshape(len(self.images), -1)

        example = self[0]
        if flatten:
            self.input_dim = example[0].shape[0]
            self.target_dim = example[1].shape[0]
        else:
            self.input_dim = example[0]
            self.target_dim = example[1]

    @property
    def shape(self):
        return self._shape

    def to_tensors(self):
        return self.images, self.images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        return img, img
