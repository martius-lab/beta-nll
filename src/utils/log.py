import argparse
import json
import pickle
import os

import numpy as np
import pandas as pd
import torch
from pathlib import Path


def train_info_to_dataframe(train_info):
    max_len = max(len(data) for data in train_info.values())

    df = pd.DataFrame(
        {name: data for name, data in train_info.items() if len(data) == max_len}
    )

    if "val_epochs" in train_info:
        index = np.array(train_info["val_epochs"]).astype(np.int)
        n_val_samples = len(train_info["val_epochs"])
        for name, data in train_info.items():
            if (
                name.startswith("val")
                and len(data) == n_val_samples
                and name != "val_epochs"
            ):
                df[name] = pd.Series(data, index=index)

    for name, data in train_info.items():
        if name.startswith("test_") and len(data) == 1:
            df[name] = pd.Series(data, index=np.array([max_len - 1], dtype=np.int))

    return df


class Logger:
    def __init__(self, root_dir, dry=False, quiet=False, flush_every=10) -> None:
        self.root_dir = Path(root_dir).absolute()
        count = 2
        dir_name = self.root_dir.name
        while self.root_dir.is_dir():
            self.root_dir = self.root_dir.with_name(f"{dir_name}_{count}")
            count += 1
        self.checkpoints_dir = self.root_dir.joinpath("checkpts")
        self.stats_dir = self.root_dir.joinpath("stats")
        self.image_dir = self.root_dir.joinpath("images")

        if not dry:
            self.root_dir.mkdir(exist_ok=True)
            self.checkpoints_dir.mkdir(exist_ok=True)
            self.stats_dir.mkdir(exist_ok=True)
            self.info_file = open(self.root_dir.joinpath("info.txt"), "a")
            self.n_writes = 0
            self.flush_every = flush_every
        else:
            self.info_file = None

        self.dry = dry
        self.quiet = quiet

    def __del__(self):
        if self.info_file:
            self.info_file.close()

    def info(self, message):
        if self.info_file:
            self.info_file.write(str(message))
            self.info_file.write("\n")
            self.n_writes += 1
            if self.n_writes == self.flush_every:
                self.info_file.flush()
                os.fsync(self.info_file)
                self.n_writes = 0

        if not self.quiet:
            print(message)

    def save_model(self, model, filename):
        path = self.checkpoints_dir.joinpath(filename).with_suffix(".pth")
        if not self.dry:
            torch.save(model.state_dict(), path)

    def save_hooks(self, hooks):
        hook_dir = self.root_dir / "hooks"
        if len(hooks) > 0 and not self.dry:
            hook_dir.mkdir(exist_ok=True)

            for hook in hooks:
                with open(hook_dir / f"{str(hook)}.pkl", "wb") as f:
                    pickle.dump(hook.state_dict(), f)

    def to_csv(self, data, filename):
        savepath = self.stats_dir.joinpath(filename).with_suffix(".csv")
        if not self.dry:
            np.savetxt(savepath, data, delimiter=",")

    def log_config(self, config):
        if not self.dry:
            if isinstance(config, argparse.Namespace):
                config = vars(config)
            with open(self.root_dir.joinpath("config.json"), "w") as fp:
                json.dump(config, fp, indent=2)

    def log_training(self, info):
        for k, v in info.items():
            self.to_csv(np.array(v), k)

    def log_training_as_dataframe(self, info):
        df = train_info_to_dataframe(info)
        if not self.dry:
            df.to_pickle(self.stats_dir / "train_info.pkl")

    def log_image(self, name, image: np.ndarray):
        if not self.dry:
            from PIL import Image

            image = np.clip((image * 255 + 0.5), 0, 255)
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
            im = Image.fromarray(image)
            self.image_dir.mkdir(exist_ok=True)
            im.save(self.image_dir / f"{name}.png")
