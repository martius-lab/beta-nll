import argparse
import json
import pathlib
import os

import pandas as pd

LOGGER = None


class Logger:
    def __init__(self, name, config, root_dir):
        self.dry = config.dry
        self.log_dir = pathlib.Path(root_dir) / name
        if not self.dry and not self.log_dir.is_dir():
            self.log_dir.mkdir()
            print(f'Logging to {self.log_dir}')

        if not config.dry:
            if isinstance(config, argparse.Namespace):
                config = vars(config)
            with open(self.log_dir / "config.json", "w") as fp:
                json.dump(config, fp, indent=2)

            self.info_file = open(self.log_dir.joinpath("info.txt"), "a")
            self.n_writes = 0
            self.flush_every = 10
        else:
            self.info_file = None

        self._data = {}
        self.write_data_every = 100

    def __del__(self):
        if self.info_file:
            self.info_file.close()

    def save_data(self):
        df = pd.DataFrame(self._data)
        if not self.dry:
            df.to_pickle(self.log_dir / "train_info.pkl")

    def last_step(self):
        self.save_data()
        self.info_file.flush()
        os.fsync(self.info_file)

    def log(self, data, step):
        for key, val in data.items():
            if key not in self._data:
                self._data[key] = {}
            self._data[key][step] = val
        if step % self.write_data_every == 0:
            self.save_data()

        msg = f'Step {step}: ' + ', '.join(f'{key}: {value}'
                                           for key, value in data.items())
        print(msg)
        if self.info_file:
            self.info_file.write(str(msg))
            self.info_file.write("\n")
            self.n_writes += 1
            if self.n_writes == 10:
                self.info_file.flush()
                os.fsync(self.info_file)
                self.n_writes = 0


def init(project, name, config, dir, tags, notes):
    global LOGGER
    LOGGER = Logger(name, config, dir)


def log(data, step):
    LOGGER.log(data, step)
