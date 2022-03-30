from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
import math


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


class Plotter:
    def __init__(self, root_path, dry=False):
        self.dry = dry
        root_dir = Path(root_path)
        self.dir = root_dir.joinpath("plots")
        if not dry:
            self.dir.mkdir(exist_ok=True)
        self.titles = {
            "losses": "Loss",
            "total_loss": "Total loss",
            "mu_losses": "Mean training loss",
            "sigma_losses": "Sigma training loss",
            "rmses": "RMSE",
            "stds": "Std",
            "elikelihoods": "Easy likelihood",
            "hlikelihoods": "Hard likelihood",
            "hmses": "Hard MSEs",
            "emses": "Easy MSEs",
            "cs": "Scheduling constant",
            "likelihoods": "Negative log-likelihood",
        }

    def _plot_dict(self, axes_iter, data, keys_to_plot, upper_percentile=99):
        plotted = []
        for k in (k for k in keys_to_plot if k in data):
            ax = next(axes_iter)
            v = data[k]
            steps = np.arange(len(v))

            ax.plot(steps, v)
            ax.plot(running_mean(v, 200), c="red")
            ax.legend([self.titles[k], "Moving average"])
            ax.set_ylabel(self.titles[k])
            ax.grid(True)

            if k == "stds":
                ax.set_yscale("log")
            else:
                limit_top = (
                    None
                    if upper_percentile is None
                    else np.percentile(v, upper_percentile)
                )
                limit_bottom = 0 if k in ["rmses", "hmses", "emses"] else None
                ax.set_ylim((limit_bottom, limit_top))
            plotted.append(k)

        return plotted

    # TODO: Refactor, it's awfully written...
    def plot_training(
        self, training_info, show=False, upper_percentile=99,
    ):
        n = 5

        rows = 3
        fig = Figure(figsize=(20, 9))
        axs = fig.subplots(rows, math.ceil(n / rows))

        training_info["rmses"] = np.sqrt(training_info["mses"])
        axs_iter = iter(axs.flatten())

        plotted_keys = []

        # Plot loss/es
        plotted_keys += self._plot_dict(
            axs_iter,
            training_info,
            ["losses", "total_loss", "mu_losses", "sigma_losses"],
            upper_percentile,
        )

        x_values = np.linspace(0, len(training_info["rmses"]))
        # Plot RMSE and Std
        plotted_keys += self._plot_dict(
            axs_iter, training_info, ["stds"], upper_percentile
        )
        # Plot training and test RMSE
        ax = next(axs_iter)
        v_rmse = training_info["rmses"]

        steps_rmse = np.arange(len(v_rmse))
        ax.plot(
            steps_rmse, training_info["rmses"], label=self.titles["rmses"],
        )
        plotted_keys += ["rmses"]

        running_trn = running_mean(training_info["rmses"], 200)
        ax.plot(
            np.arange(len(running_trn)),
            running_trn,
            label="RMSE running mean",
            linestyle="dashed",
            c="red",
        )

        if "val_mses" in training_info:
            training_info["val_rmses"] = np.sqrt(training_info["val_mses"])
            v_eval = training_info["val_rmses"]
            steps_val = training_info["val_epochs"]
            ax.plot(
                steps_val, v_eval, label="Validation RMSE", c="orange",
            )
            plotted_keys += ["val_rmses"]
            limit_top = max(
                np.percentile(v_rmse, upper_percentile),
                np.percentile(v_eval, upper_percentile),
            )
            ax.set_ylim((None, limit_top))

        ax.set_title("RMSE")
        ax.grid(True)
        ax.legend()

        # Plot likelihoods and mses
        for pair, title in zip(
            [["elikelihoods", "hlikelihoods"], ["emses", "hmses"]],
            ["Negative log-likelihood", "Mean Squared Error",],
        ):
            e, h = pair
            if not set(pair).issubset(training_info.keys()):
                continue
            ax = next(axs_iter)
            v_e = training_info[e]
            v_h = training_info[h]

            ax.plot(np.arange(len(v_e)), training_info[e], label=self.titles[e])
            ax.plot(np.arange(len(v_h)), training_info[h], label=self.titles[h])
            limit_top = max(
                np.nanpercentile(v_e, upper_percentile),
                np.nanpercentile(v_h, upper_percentile),
            )
            limit_top = (
                None if np.isinf(limit_top) or np.isnan(limit_top) else limit_top
            )
            ax.set_ylim((None, limit_top))
            ax.set_title(title)
            ax.grid(True)
            ax.legend()
            plotted_keys += pair

        fig.tight_layout()

        if not self.dry:
            fig.savefig(self.dir.joinpath("training.pdf"))
        if show:
            plt.show()

    def plot_data_for_ds3(self, X, Y, x, y_mean, y_var, show=False, ax=None):
        if ax is None:
            ax = plt
        ax.plot(x, y_mean)
        ax.fill_between(x, y_mean - np.sqrt(y_var), y_mean + np.sqrt(y_var), alpha=0.2)
        ax.plot(X, Y, ".", color="red", alpha=0.15)
        ax.plot(x, x * np.sin(x), "-", color="black")
        if not self.dry:
            plt.savefig(self.dir.joinpath("data3.pdf"))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_data(
        self,
        X,
        Y,
        x,
        y_mean,
        y_var,
        show=False,
        ground_truth=None,
        filename=None,
        ax=None,
    ):
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.squeeze()
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.squeeze()
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.squeeze()
        if y_mean.ndim == 2 and y_mean.shape[1] == 1:
            y_mean = y_mean.squeeze()
        if y_var.ndim == 2 and y_var.shape[1] == 1:
            y_var = y_var.squeeze()

        if ax is None:
            fig = Figure()
            ax = fig.subplots()
        else:
            fig = None
        if filename is None:
            filename = "data.pdf"
        ax.plot(x, y_mean)
        ax.fill_between(x, y_mean - np.sqrt(y_var), y_mean + np.sqrt(y_var), alpha=0.2)
        ax.plot(X, Y, ".", color="red", alpha=0.15)
        if ground_truth is not None:
            ax.plot(
                X, ground_truth, ".", color="black", alpha=0.2,
            )

        if not self.dry and fig is not None:
            fig.savefig(self.dir.joinpath(filename))
        if show:
            plt.show()
