from pathlib import Path

import numpy as np
import torch

DATA_FOLDER = Path(__file__).parent.parent.parent.absolute().joinpath("data")


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, noise, labels):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=-1)
        if Y.ndim == 1:
            Y = np.expand_dims(Y, axis=-1)
        if noise is not None and noise.ndim == 1:
            noise = np.expand_dims(noise, axis=-1)
        if labels is not None and labels.ndim == 1:
            labels = np.expand_dims(labels, axis=-1)
        self.X = X
        self.noise = noise
        self.Y = Y
        self.pts = Y + noise
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.pts[idx], self.labels[idx]

    def save(self, path):
        np.savez(path, X=self.X, Y=self.Y, noise=self.noise, labels=self.labels)

    @classmethod
    def get_dataset(cls, ds, size, seed=None):
        d = {
            1: ToyDataset._make_dataset1,
            2: ToyDataset._make_dataset2,
            3: ToyDataset._make_dataset3,
            4: ToyDataset._make_dataset4,
            5: ToyDataset._make_dataset5,
            6: ToyDataset._make_dataset6,
            7: ToyDataset._make_dataset7,
            8: ToyDataset._make_dataset8,
            9: ToyDataset._make_dataset9,
            10: ToyDataset._make_dataset10,
            11: ToyDataset._make_dataset11,
            12: ToyDataset._make_dataset12,
            13: ToyDataset._make_dataset13,
            14: ToyDataset._make_dataset14,
            15: ToyDataset._make_dataset15,
        }
        data = d[ds](n=size, seed=seed)
        X = data[-4]
        Y = data[-3]
        noise = data[-2]
        if noise is None:
            noise = np.zeros_like(Y)
        labels = data[-1]
        if labels is None:
            labels = np.zeros_like(Y)
        return cls(X, Y, noise, labels)

    ##########################################

    @classmethod
    def load_data(cls, n_ds, data_fn=None, persist=True):
        if data_fn is None:
            data_fn = lambda seed: cls.get_dataset(n_ds, size=1000, seed=seed)

        save_train = False
        save_test = False
        if persist:
            data_dir = DATA_FOLDER.joinpath(str(n_ds))
            data_dir.mkdir(exist_ok=True)

            train_path = data_dir.joinpath("train.npz")
            test_path = data_dir.joinpath("test.npz")

            if train_path.exists():
                train = np.load(train_path, allow_pickle=True)
                train = cls(train["X"], train["Y"], train["noise"], train["labels"])
            else:
                train = data_fn(seed=0)
                save_train = True

            if test_path.exists():
                test = np.load(test_path, allow_pickle=True)
                test = cls(test["X"], test["Y"], test["noise"], test["labels"])
            else:
                test = data_fn(seed=1)
                save_test = True
        else:
            train = data_fn(seed=0)
            test = data_fn(seed=1)

        if save_train:
            train.save(train_path)
        if save_test:
            test.save(test_path)

        return train, test

    @staticmethod
    def _make_dataset(xs, ys, p_x, std_eps, n, verbose=False, seed=None):
        """Creates a dataset where each noisy output y is assigned to one cluster identified by the input x"""
        if seed is None:
            seed = np.random.randint(2 ** 32 - 1)
        rng = np.random.RandomState(seed)

        X = rng.choice(xs, size=n, replace=True, p=p_x)

        min_mses = []

        Y = np.zeros_like(X)
        eps = rng.randn(*Y.shape)
        for x, y, std in zip(xs, ys, std_eps):
            eps[X == x] *= std
            Y[X == x] = y + eps[X == x]

            min_mses.append(eps[X == x] ** 2)

        if verbose:
            print(f"Min MSE: {np.mean(np.concatenate(min_mses)):.4f}")

        return X, Y, eps

    @staticmethod
    def _make_dataset1(n=1000, seed=None):
        """Dataset with same mean, but different noise level for each x"""
        xs = np.array([-4, -2, 0, 2, 4.0])
        ys = np.array([0, 0, 0, 0, 0])
        std_eps = np.array([1, 0.1, 1, 0.5, 2])
        X, Y, eps = ToyDataset._make_dataset(
            xs, ys, p_x=np.array([1 / 5] * 5), std_eps=std_eps, n=n, seed=seed
        )
        return xs, ys, std_eps, X, Y

    @staticmethod
    def _make_dataset2(n=1000, seed=None):
        """Dataset with different means and noise levels for each x"""
        xs = np.array([-4, -2, 0, 2, 4.0])
        ys = np.array([4, 0, -2, 3, 2])
        std_eps = np.array([1, 0.1, 1, 0.5, 2])
        X, Y, eps = ToyDataset._make_dataset(
            xs, ys, p_x=np.array([1 / 5] * 5), std_eps=std_eps, n=n, seed=seed
        )
        return xs, ys, std_eps, X, Y

    @staticmethod
    def _make_dataset3(n=500, seed=None):
        """Toy data from Detlefsen et al: Reliable training and estimation of variance networks"""
        X = np.random.uniform(0, 10, size=n)
        Y = X * np.sin(X) + 0.3 * np.random.randn(n) + 0.3 * X * np.random.randn(n)
        return X, Y, None, None

    @staticmethod
    def _make_dataset4(
        mean=0, std=1, outlier_err=1, outlier_size=100, n=1000, seed=None
    ):
        X = np.random.uniform(-5, 5, size=n)

        n -= outlier_size
        outlier_batch = outlier_size // 2

        # "Standard data"
        errs = std * np.random.rand(n) * np.random.choice((-1, 1), n)
        Y = mean + errs

        # "Lower outliers"
        lower_errs = outlier_err * np.random.rand(outlier_batch)
        lower_outliers = mean - 3 * std - lower_errs

        # "Upper outliers"
        upper_errs = outlier_err * np.random.rand(outlier_batch)
        upper_outliers = mean + 3 * std + upper_errs

        Y = np.concatenate((Y, lower_outliers, upper_outliers))
        np.random.shuffle(Y)

        return X, Y

    @staticmethod
    def _make_dataset5(
        mean=0, std=1, outlier_err=1, outlier_size=200, n=1000, seed=None
    ):
        X = np.random.uniform(0, 10, size=n)

        noise = ToyDataset._sample_with_outliers(
            mean, std, outlier_err, outlier_size, n
        )

        return X, X * (0.4 * np.sin(X)), X * 0.3 * noise

    @staticmethod
    def _sample_with_outliers(
        mean=0,
        std=1,
        outlier_err=1,
        outlier_size=200,
        up=True,
        down=True,
        center=True,
        n=1000,
        outlier_factor=1.5,
        seed=None,
    ):
        if seed is None:
            seed = np.random.randint(2 ** 32 - 1)
        rng = np.random.RandomState(seed)

        center_size = n - outlier_size

        # Avoid odd number division by 2
        upper_size = outlier_size // 2
        lower_size = outlier_size - upper_size

        lower_outliers, upper_outliers = np.array([]), np.array([])

        errs = np.array([])
        if center:
            # "Standard data"
            errs = (
                std
                * rng.normal(size=center_size)
                * rng.choice((-1, 1), size=center_size)
            )

        if down:
            # "Lower outliers"
            lower_errs = outlier_err * rng.normal(size=lower_size)
            lower_outliers = -outlier_factor * std - lower_errs

        if up:
            # "Upper outliers"
            upper_errs = outlier_err * rng.normal(size=upper_size)
            upper_outliers = outlier_factor * std + upper_errs

        return np.concatenate((errs, lower_outliers, upper_outliers))

    @staticmethod
    def _make_dataset6(n=1000, seed=None):
        xs = np.array([-4, -2, 0, 2, 4.0])
        ys = np.array([4, 0, -2, 3, 2])
        std_eps = np.array([1, 3, 1, 0.5, 2])
        p_x = np.array([1 / 5] * 5)

        X = np.random.choice(xs, size=n, replace=True, p=p_x)

        Y = np.zeros_like(X)

        y_noise = []
        for x, mean_y, std in zip(xs, ys, std_eps):
            xs_size = np.sum(X == x)
            outliers_size = np.sum(np.random.uniform(size=xs_size) >= 0.45)

            s_y = ToyDataset._sample_with_outliers(
                mean=mean_y,
                std=std,
                outlier_err=1,
                outlier_size=outliers_size,
                n=xs_size,
                outlier_factor=4,
                seed=seed,
            )
            y_noise.append(mean_y + s_y)
            Y[X == x] = s_y

        return X, Y, y_noise

    @staticmethod
    def _make_dataset7(mean=0, std=1, outlier_err=1, n=1000, seed=None):
        xs = [
            np.random.uniform(0, 3, size=300),
            np.random.uniform(3, 7, size=400),
            np.random.uniform(7, 10, size=300),
        ]
        outlier_sizes = [100, 400, 100]
        noise = []
        for X, center, factor, out_size in zip(
            xs, [True, False, True], [1.5, 4, 1.5], outlier_sizes
        ):
            pts = ToyDataset._sample_with_outliers(
                mean,
                std,
                outlier_err,
                outlier_factor=factor,
                outlier_size=out_size,
                n=X.shape[0],
                center=center,
                seed=seed,
            )
            noise.append(pts + mean)

        X = np.concatenate(xs)
        noise = np.concatenate(noise)

        return X, X * (0.4 * np.sin(X)), X * 0.3 * noise

    @staticmethod
    def _make_dataset8(n=1000, seed=None):
        n1 = n // 3
        n2 = n // 3
        n3 = n - n1 - n2
        assert n1 + n2 + n3 == n

        noise = ToyDataset._sample_with_outliers(
            0, 0.01, outlier_size=0, n=n, seed=seed
        )
        X = np.linspace(0, 12, num=n)

        f = np.repeat([0.5, 1, 0.5], [n1, n2, n3])
        c = np.repeat([0.2, 0.4, 0.2], [n1, n2, n3])
        Y = c * np.sin(2 * np.pi * f * X)
        labels = np.repeat([0, 1, 0], [n1, n2, n3])

        return (
            X,
            Y,
            noise,
            labels,
        )

    @staticmethod
    def _make_dataset9(n=1000, seed=None):
        n1 = n // 3
        n2 = n // 3
        n3 = n - n1 - n2
        assert n1 + n2 + n3 == n

        X = np.linspace(0, 12, num=n)
        noise = ToyDataset._sample_with_outliers(
            0,
            std=(0.07 * np.abs(np.sin(X))),
            outlier_size=0,
            n=n,
            up=False,
            down=False,
            seed=seed,
        )

        f = np.repeat([0.5, 1, 0.5], [n1, n2, n3])
        c = np.repeat([0.2, 0.4, 0.2], [n1, n2, n3])
        Y = c * np.sin(2 * np.pi * f * X)
        labels = np.repeat([0, 1, 0], [n1, n2, n3])
        # import matplotlib.pyplot as plt

        # plt.plot(X, Y + noise)
        # plt.show()

        return (
            X,
            Y,
            noise,
            labels,
        )

    @staticmethod
    def _make_dataset10(n=1000, seed=None):
        """Like dataset 8, but only the hard part"""
        n1 = n // 3
        n2 = n // 3
        n3 = n - n1 - n2
        assert n1 + n2 + n3 == n

        noise = ToyDataset._sample_with_outliers(
            0, 0.01, outlier_size=0, n=n, seed=seed
        )
        X = np.linspace(0, 12, num=n)

        f = np.repeat([0.5, 1, 0.5], [n1, n2, n3])
        c = np.repeat([0.2, 0.4, 0.2], [n1, n2, n3])
        Y = c * np.sin(2 * np.pi * f * X)
        labels = np.repeat([0, 1, 0], [n1, n2, n3])

        return (
            X[n1 : n1 + n2],
            Y[n1 : n1 + n2],
            noise[n1 : n1 + n2],
            labels[n1 : n1 + n2],
        )

    @staticmethod
    def _make_dataset11(n=1000, seed=None):
        """Like dataset 8, but the hard part extended to [0, 12]"""
        noise = ToyDataset._sample_with_outliers(
            0, 0.01, outlier_size=0, n=n, seed=seed
        )
        X = np.linspace(0, 12, num=n)

        f = np.repeat([1], [n])
        c = np.repeat([0.4], [n])
        Y = c * np.sin(2 * np.pi * f * X)
        labels = np.repeat([1], [n])

        return (
            X,
            Y,
            noise,
            labels,
        )

    @staticmethod
    def _make_dataset12(n=1000, seed=None):
        """Like dataset 11, but the range shifted to [-6, 6]"""
        noise = ToyDataset._sample_with_outliers(
            0, 0.01, outlier_size=0, n=n, seed=seed
        )
        X = np.linspace(-6, 6, num=n)

        f = np.repeat([1], [n])
        c = np.repeat([0.4], [n])
        Y = c * np.sin(2 * np.pi * f * X)
        labels = np.repeat([1], [n])

        return (
            X,
            Y,
            noise,
            labels,
        )

    @staticmethod
    def _make_dataset13(n=1000, seed=None):
        """Like dataset 11, but the range shifted to mean zero and normalized"""
        noise = ToyDataset._sample_with_outliers(
            0, 0.01, outlier_size=0, n=n, seed=seed
        )
        X = np.linspace(-6, 6, num=n)

        f = np.repeat([1], [n])
        c = np.repeat([0.4], [n])
        Y = c * np.sin(2 * np.pi * f * X)
        labels = np.repeat([1], [n])

        X = X / np.std(X)

        return (
            X,
            Y,
            noise,
            labels,
        )

    @staticmethod
    def _make_dataset14(n=1000, seed=None):
        """Like dataset 11, but rotated by 20 degrees"""
        noise = ToyDataset._sample_with_outliers(
            0, 0.01, outlier_size=0, n=n, seed=seed
        )
        X = np.linspace(0, 12, num=n)

        f = np.repeat([1], [n])
        c = np.repeat([0.4], [n])
        Y = c * np.sin(2 * np.pi * f * X)
        labels = np.repeat([1], [n])

        angle = np.radians(20)
        xy = np.stack((X, Y), axis=0)
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        xy_rot = rot.dot(xy)
        X = xy_rot[0]
        Y = xy_rot[1]

        return (
            X,
            Y,
            noise,
            labels,
        )

    @staticmethod
    def _make_dataset15(n=1000, seed=None):
        """Like dataset 14, but shifted to range [-6, 6]"""
        noise = ToyDataset._sample_with_outliers(
            0, 0.01, outlier_size=0, n=n, seed=seed
        )
        X = np.linspace(-6, 6, num=n)

        f = np.repeat([1], [n])
        c = np.repeat([0.4], [n])
        Y = c * np.sin(2 * np.pi * f * X)
        labels = np.repeat([1], [n])

        angle = np.radians(20)
        xy = np.stack((X, Y), axis=0)
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        xy_rot = rot.dot(xy)
        X = xy_rot[0]
        Y = xy_rot[1]

        return (
            X,
            Y,
            noise,
            labels,
        )
