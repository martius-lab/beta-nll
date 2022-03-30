import pathlib
import pickle

import numpy as np
import torch

from src.data.utils import split

DATA_NAME = "fpp_rollouts_apex_1.pkl"
DEFAULT_DATA_DIR = "./data/"

SENSOR_INFO_PNP = {
    "grip_pos": [0, 1, 2],
    "object_pos": [3, 4, 5],
    "object_rel_pos": [6, 7, 8],
    "gripper_state": [9, 10],
    "object_rot": [11, 12, 13],
    "object_velp": [14, 15, 16],
    "object_velr": [17, 18, 19],
    "grip_velp": [20, 21, 22],
    "gripper_vel": [23, 24],
}


class FPPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        use_diff_as_target=True,
        train_split=0.7,
        test_split=0.15,
        standardize_inputs=False,
        standardize_targets=True,
        obs_noise_level=None,
        data_dir=None,
        only_object_noise=False,
    ) -> None:
        self.use_diff_as_target = use_diff_as_target
        self.only_object_noise = only_object_noise

        if data_dir is None:
            data_dir = pathlib.Path(DEFAULT_DATA_DIR)
        else:
            data_dir = pathlib.Path(data_dir)

        with open(data_dir / DATA_NAME, "rb") as f:
            data = pickle.load(f)

        train_data, val_data, test_data = split(
            data, train_split, test_size=test_split, shuffle=True, seed=42
        )

        if obs_noise_level is not None:

            def make_obs_and_noises(data):
                obs = [rollout["observations"] for rollout in data]
                obs = np.concatenate(obs, axis=0)
                noises = noise_rng.randn(2, *obs.shape)
                return obs, noises

            noise_rng = np.random.RandomState(333)
            obs, _ = make_obs_and_noises(data)
            self.obs_noise_std = np.std(obs, axis=0) * obs_noise_level

            _, train_noises = make_obs_and_noises(train_data)
            if val_data is not None:
                _, val_noises = make_obs_and_noises(val_data)
            else:
                val_noises = None
            if test_data is not None:
                _, test_noises = make_obs_and_noises(test_data)
            else:
                test_noises = None
        else:
            train_noises = None
            val_noises = None
            test_noises = None

        self._train_data = self._init_data(train_data, train_noises)
        self._val_data = self._init_data(val_data, val_noises)
        self._test_data = self._init_data(test_data, test_noises)

        if standardize_inputs:
            mean = self._train_data[0].mean(axis=0)
            std = self._train_data[0].std(axis=0)
            self._train_data[0] = (self._train_data[0] - mean) / std
            if val_data is not None:
                self._val_data[0] = (self._val_data[0] - mean) / std
            if test_data is not None:
                self._test_data[0] = (self._test_data[0] - mean) / std
            self.input_mean = mean
            self.input_std = std
        else:
            self.input_mean = None
            self.input_std = None

        if standardize_targets:
            self.target_mean = self._train_data[1].mean(axis=0)
            self.target_std = self._train_data[1].std(axis=0)
            self._train_data[1] = (
                self._train_data[1] - self.target_mean
            ) / self.target_std
        else:
            self.target_mean = None
            self.target_std = None

    @property
    def train_data(self):
        dataset = self._init_tensor_dataset(self._train_data)
        if self.target_mean is not None:
            dataset.target_mean = self.target_mean
            dataset.target_std = self.target_std
        return dataset

    @property
    def val_data(self):
        return self._init_tensor_dataset(self._val_data)

    @property
    def test_data(self):
        return self._init_tensor_dataset(self._test_data)

    def _init_tensor_dataset(self, data):
        if data is None:
            return None
        inps = torch.from_numpy(data[0]).float()
        tgts = torch.from_numpy(data[1]).float()
        return torch.utils.data.TensorDataset(inps, tgts)

    def _init_data(self, rollouts, noises=None):
        if rollouts is None:
            return None
        inputs = []
        targets = []
        idx = 0
        for rollout in rollouts:
            observations = rollout["observations"]
            next_observations = rollout["next_observations"]

            if noises is not None:
                obs_noise = noises[0, idx : idx + len(observations)]
                observations = add_observation_noise(
                    observations, obs_noise, self.obs_noise_std, self.only_object_noise
                )
                next_obs_noise = noises[1, idx : idx + len(observations)]
                next_observations = add_observation_noise(
                    next_observations,
                    next_obs_noise,
                    self.obs_noise_std,
                    self.only_object_noise,
                )

            # Get rid of rollout goal
            observations = observations[..., :-3]
            next_observations = next_observations[..., :-3]

            actions = rollout["actions"]
            inp = np.concatenate((observations, actions), axis=1)

            target = self._get_target(observations, next_observations)

            inputs.append(inp)
            targets.append(target)
            idx += len(observations)

        self.input_dim = inp.shape[1]
        self.target_dim = target.shape[1]

        return [np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)]

    def _get_target(self, obs, n_obs):
        target = None
        if self.use_diff_as_target:
            target = (
                n_obs[:, SENSOR_INFO_PNP["object_pos"]]
                - obs[:, SENSOR_INFO_PNP["object_pos"]]
            )
        else:
            target = n_obs[:, SENSOR_INFO_PNP["object_pos"]]

        return target

    @staticmethod
    def _get_observations_std(rollouts):
        observations = [rollout["observations"] for rollout in rollouts]
        return np.std(observations, axis=0)


def add_observation_noise(obs, noises, stds, only_object_noise=False):
    """Add noise to observations

    `noises`: Standard normal noise of same shape as `obs`
    `stds`: Standard deviation per dimension of `obs` to scale noise with
    """
    assert obs.shape == noises.shape
    idxs_object_pos = SENSOR_INFO_PNP["object_pos"]
    agent_vel = obs[..., SENSOR_INFO_PNP["grip_velp"]]
    obs = obs.copy()

    if only_object_noise:
        obs[..., idxs_object_pos] += (
            noises[..., idxs_object_pos] * stds[..., idxs_object_pos]
        )
    else:
        obs += noises * stds

    # Recompute relative position
    obs[..., SENSOR_INFO_PNP["object_rel_pos"]] = (
        obs[..., SENSOR_INFO_PNP["object_pos"]] - obs[..., SENSOR_INFO_PNP["grip_pos"]]
    )

    # Recompute relative speed: first add old agent velocity to get noisy
    # object velocity, then subtract noisy agent velocity to get correct
    # relative speed between noisy measurements
    obs[..., SENSOR_INFO_PNP["object_velp"]] = (
        obs[..., SENSOR_INFO_PNP["object_velp"]]
        + agent_vel
        - obs[..., SENSOR_INFO_PNP["grip_velp"]]
    )

    return obs
