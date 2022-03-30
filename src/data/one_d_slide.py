"""Dataset of 1D-Slide environment

The task in the 1D-Slide environment is for the agent to shoot an object into
a target zone. Both agent an object can only move left and right on a
one-dimensional line. As the agent can not cross past the center of the line,
it has to hit the object at the appropriate speed.

Observation Space:
- 0: Position of agent
- 1: Velocity of agent
- 2: Position of object
- 3: Velocity of object
"""
import pathlib

import numpy as np
import torch

DEFAULT_DATA_DIR = pathlib.Path("./data")

DATASETS = {
    # 2000 rollouts of random agent
    "random2k": (
        "train_memory_2k_random_agent.npy",
        "val_memory_2k_random_agent.npy",
        "test_memory_2k_random_agent.npy",
    ),
}


def get_data_paths(variant, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_DATA_DIR
    else:
        base_dir = pathlib.Path(base_dir)

    return (
        base_dir / DATASETS[variant][0],
        base_dir / DATASETS[variant][1],
        base_dir / DATASETS[variant][2],
    )


def make_one_d_slide_dataset(path, **kwargs):
    return _dataset_from_stored_memory(path, **kwargs)


def _dataset_from_stored_memory(memory_path, *args, **kwargs):
    data = np.load(memory_path, allow_pickle=True)
    size = data[1]
    buffers = {key: value[:size] for key, value in data[0].items()}

    return BufferDataset(buffers, *args, **kwargs)


class BufferDataset(torch.utils.data.Dataset):
    """Dataset loading directly from buffer dict"""

    def __init__(
        self,
        buffers,
        use_goal_diff_as_target=True,
        only_random_actions=True,
        only_contacts=False,
        only_movements=False,
        movement_epsilon=1e-5,
        goal_idx=2,
        obs_noise_level=None,
        only_goal_noise=False,
    ):
        required_keys = {"s", "a", "ag"}
        if only_random_actions:
            required_keys.add("rand_a")
        if only_contacts:
            required_keys.add("contact")
        for key in required_keys:
            if key not in buffers:
                raise ValueError(f"Required key {key} not in buffer")
        self.goal_idx = goal_idx

        n_episodes, episode_len = buffers["s"].shape[:2]
        buffers["ag_next"] = np.roll(buffers["ag"], -1, axis=1)

        selection = np.ones((n_episodes, episode_len,), dtype=bool,)
        selection[:, -1] = False  # Filter last state of each episode

        if only_random_actions:
            selection &= buffers["rand_a"].astype(bool).squeeze(axis=-1)
        if only_contacts:
            selection &= buffers["contact"].astype(bool).squeeze(axis=-1)
        if only_movements:
            goal_diff = buffers["ag_next"] - buffers["ag"]
            movement_norm = np.linalg.norm(goal_diff, ord=2, axis=-1)
            selection &= movement_norm >= movement_epsilon

        self._idxs = np.nonzero(selection.reshape(-1))[0]
        buffers = {
            key: value.reshape(-1, *value.shape[2:]) for key, value in buffers.items()
        }
        self._use_goal_diff_as_target = use_goal_diff_as_target
        self._only_goal_noise = only_goal_noise

        if obs_noise_level is not None:
            noise_rng = np.random.RandomState(333)
            obs_noises = noise_rng.randn(2, *buffers["s"].shape)
            self.obs_noise_std = np.std(buffers["s"], axis=0) * obs_noise_level
        else:
            obs_noises = None

        self.inputs, self.targets = self._init_data(buffers, obs_noises)

    def _init_data(self, buffers, obs_noises=None):
        buffer_idx = self._idxs[:]

        s = buffers["s"][buffer_idx].astype(np.float32)
        if obs_noises is not None:
            noise = obs_noises[0, buffer_idx]
            noise_next = obs_noises[1, buffer_idx]
            goal_std = self.obs_noise_std[self.goal_idx]
            goal_slice = slice(self.goal_idx, self.goal_idx + 1)
            if self._only_goal_noise:
                s = s.copy()[:, goal_slice] + noise[:, goal_slice] * goal_std
            else:
                s = s + noise * self.obs_noise_std

        a = buffers["a"][buffer_idx].astype(np.float32)
        inp = np.concatenate((s, a), axis=-1)

        ag = buffers["ag"][buffer_idx]
        ag_next = buffers["ag_next"][buffer_idx]

        if obs_noises is not None:
            ag = ag + noise[:, goal_slice] * goal_std
            ag_next = ag_next + noise_next[:, goal_slice] * goal_std

        if self._use_goal_diff_as_target:
            target = (ag_next - ag).astype(np.float32)
        else:
            target = ag_next.astype(np.float32)

        return inp, target

    def standardize_inputs(self, mean=None, std=None):
        if mean is None and std is None:
            mean = self.inputs.mean(axis=0)
            std = self.inputs.std(axis=0)

        self.inputs = (self.inputs - mean) / std

        return mean, std

    def standardize_targets(self, mean=None, std=None):
        if mean is None and std is None:
            mean = self.targets.mean()
            std = self.targets.std()
            self.target_mean = mean
            self.target_std = std

        self.targets = (self.targets - mean) / std

        return mean, std

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]
