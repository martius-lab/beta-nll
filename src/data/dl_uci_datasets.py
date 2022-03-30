"""Download some of the UCI datasets

Adapted from Stirn et al, https://github.com/astirn/variational-variance
under MIT License.

We only use this to download carbon, energy, naval, superconductivity, wine-white
"""
import sys
import os
import glob
import zipfile
import warnings
import numpy as np
import pandas as pd
from urllib import request
from sklearn.model_selection import train_test_split


def trim_eol_whitespace(data_file):
    with open(data_file, "r") as f:
        lines = f.readlines()
    lines = [line.replace(" \n", "\n") for line in lines]
    with open(data_file, "w") as f:
        f.writelines(lines)


def decimal_comma_to_decimal_point(data_file):
    with open(data_file, "r") as f:
        lines = f.readlines()
    lines = [line.replace(",", ".") for line in lines]
    with open(data_file, "w") as f:
        f.writelines(lines)


REGRESSION_DATA = {
    "boston": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        "dir_after_unzip": None,
        "data_file": "housing.data",
        "parse_args": {"sep": " ", "header": None, "skipinitialspace": True},
        "target_cols": [-1],
    },
    "carbon": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00448/carbon_nanotubes.csv",
        "dir_after_unzip": None,
        "data_file": "carbon_nanotubes.csv",
        "formatter": decimal_comma_to_decimal_point,
        "parse_args": {"sep": ";"},
        "target_cols": [-1, -2, -3],
    },
    "concrete": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        "dir_after_unzip": None,
        "data_file": "Concrete_Data.xls",
        "parse_args": dict(),
        "target_cols": [-1],
    },
    "energy": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "dir_after_unzip": None,
        "data_file": "ENB2012_data.xlsx",
        "parse_args": dict(),
        "target_cols": [-1, -2],
    },
    "naval": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
        "dir_after_unzip": "UCI CBM Dataset",
        "data_file": "data.txt",
        "parse_args": {"sep": " ", "header": None, "skipinitialspace": True},
        "target_cols": [-1, -2],
    },
    "power plant": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
        "dir_after_unzip": "CCPP",
        "data_file": "Folds5x2_pp.xlsx",
        "parse_args": dict(),
        "target_cols": [-1],
    },
    "protein": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        "dir_after_unzip": None,
        "data_file": "CASP.csv",
        "parse_args": dict(),
        "target_cols": [1],
    },
    "superconductivity": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip",
        "dir_after_unzip": None,
        "data_file": "train.csv",
        "parse_args": dict(),
        "target_cols": [-1],
    },
    "wine-red": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "dir_after_unzip": None,
        "data_file": "winequality-red.csv",
        "parse_args": {"sep": ";"},
        "target_cols": [-1],
    },
    "wine-white": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        "dir_after_unzip": None,
        "data_file": "winequality-white.csv",
        "parse_args": {"sep": ";"},
        "target_cols": [-1],
    },
    "yacht": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        "dir_after_unzip": None,
        "data_file": "yacht_hydrodynamics.data",
        "formatter": trim_eol_whitespace,
        "parse_args": {"sep": " ", "header": None, "skipinitialspace": True},
        "target_cols": [-1],
    },
    "year": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
        "dir_after_unzip": None,
        "data_file": "YearPredictionMSD.txt",
        "parse_args": dict(),
        "target_cols": [1],
    },
}


def download(key, base_dir, force_download=False):
    data_dir = os.path.join(base_dir, key)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    file = os.path.join(data_dir, REGRESSION_DATA[key]["url"].split("/")[-1])
    if os.path.exists(file) and force_download:
        os.remove(file)
    elif os.path.exists(file) and not force_download:
        print(file.split(os.sep)[-1], "already exists.")
        return

    print("Downloading", file.split(os.sep)[-1])
    request.urlretrieve(REGRESSION_DATA[key]["url"], file)


def download_all(force_download=False):

    # make data directory if it doesn't yet exist
    if not os.path.exists("data"):
        os.mkdir("data")

    # download all regression data experiments
    for key in REGRESSION_DATA.keys():
        download(key, "data")
    print("Downloads complete!")


def load_data(data_dir, dir_after_unzip, data_file, parse_args, **kwargs):

    # save the base data directory as the save directory, since data_dir might be modified below
    save_dir = data_dir

    # find any zip files
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    assert len(zip_files) <= 1

    # do we need to unzip?
    if len(zip_files) or dir_after_unzip is not None:

        # unzip it
        with zipfile.ZipFile(zip_files[0], "r") as f:
            f.extractall(data_dir)

        # update data directory if required
        if dir_after_unzip is not None:
            data_dir = os.path.join(data_dir, dir_after_unzip)

    # correct formatting issues if necessary
    if "formatter" in kwargs.keys() and kwargs["formatter"] is not None:
        kwargs["formatter"](os.path.join(data_dir, data_file))

    # process files according to type
    if os.path.splitext(data_file)[-1] in {".csv", ".data", ".txt"}:
        df = pd.read_csv(os.path.join(data_dir, data_file), **parse_args)
    elif os.path.splitext(data_file)[-1] in {".xls", ".xlsx"}:
        df = pd.read_excel(os.path.join(data_dir, data_file))
    else:
        warnings.warn("Type Not Supported: " + data_file)
        return

    # convert to numpy arrays
    xy = df.dropna(axis=1, how="all").to_numpy(dtype=np.float32)

    save_dir = os.path.join(save_dir, "data")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    np.savetxt(os.path.join(save_dir, "data.txt"), xy)
    D = xy.shape[1]

    y_indices = kwargs["target_cols"]
    x_indices = list(range(D))
    y_indices = [D + i if i < 0 else i for i in y_indices]
    for i in y_indices:
        x_indices.pop(i)

    np.savetxt(
        os.path.join(save_dir, "index_features.txt"), np.array(x_indices), fmt="%d"
    )
    np.savetxt(
        os.path.join(save_dir, "index_target.txt"), np.array(y_indices), fmt="%d"
    )
    n_splits = 20
    np.savetxt(os.path.join(save_dir, "n_splits.txt"), np.array([n_splits]), fmt="%d")

    indices = np.arange(len(xy))[:, None]
    for idx in range(n_splits):
        train_idxs, test_idxs = train_test_split(
            indices, test_size=0.1, shuffle=True, random_state=6 + idx
        )
        np.savetxt(
            os.path.join(save_dir, f"index_train_{idx}.txt"), train_idxs, fmt="%d"
        )
        np.savetxt(os.path.join(save_dir, f"index_test_{idx}.txt"), test_idxs, fmt="%d")


if __name__ == "__main__":
    if len(sys.argv) == 0:
        print("Supply dataset to download")
        sys.exit()
    key = sys.argv[1]
    if len(sys.argv) >= 3:
        path = sys.argv[2]
    else:
        path = "./data"

    download(key, path)

    load_data(data_dir=os.path.join(path, key), **REGRESSION_DATA[key])
    print("Processing complete!")
