import math

import numpy as np
from sklearn.model_selection import train_test_split


def split(
    data, train_size=None, val_size=None, test_size=None, shuffle=True, seed=None
):
    """Split data into train-val-test

    If test_size is `None`, test split will be empty.
    Assume `data` is a list of numpy arrays
    """
    if train_size is None and val_size is None:
        raise ValueError("At least one of train_size, val_size must be specified")

    assert train_size is None or 0.0 <= train_size <= 1.0
    assert val_size is None or 0.0 <= val_size <= 1.0
    assert test_size is None or 0.0 <= test_size <= 1.0
    test_size = test_size or 0.0
    train_size = train_size or 1.0 - val_size - test_size
    val_size = val_size or 1.0 - train_size - test_size

    n_test = math.ceil(len(data) * test_size)
    n_val = math.ceil(len(data) * val_size)
    n_train = len(data) - n_test - n_val

    if shuffle:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(data))
    else:
        indices = np.arange(len(data))

    indices_train = indices[:n_train]
    indices_val = indices[n_train : n_train + n_val]
    indices_test = indices[n_train + n_val :]

    return (
        [data[idx] for idx in indices_train],
        [data[idx] for idx in indices_val] if val_size > 0 else None,
        [data[idx] for idx in indices_test] if test_size > 0 else None,
    )


def split_in_chunks(
    data,
    chunk_size,
    train_size=None,
    val_size=None,
    test_size=None,
    shuffle=True,
    seed=None,
):
    """Split data into train-test, where chunks of data are held together

    Assume `data` is a list of numpy arrays
    """
    chunks = []
    for rollout in data:
        for idx in range(0, len(rollout), chunk_size):
            chunk = rollout[idx : idx + chunk_size]
            chunks.append(chunk)

    return split(chunks, train_size, val_size, test_size, shuffle, seed)
