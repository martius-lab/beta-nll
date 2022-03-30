import itertools
import math
import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch

from src import data
from src.models import utils as nnutils
from src.models.networks import MLP
from src.train import parser as opts_parser
from src.trainers import utils as tutils
from src.utils.log import Logger

parser = ArgumentParser()
parser.add_argument(
    "-n", "--dry", help="Set if running only for sanity check", action="store_true"
)
parser.add_argument("--device", help="Device name for training", default="cpu")
parser.add_argument("--seed", help="Random seed", type=int)
parser.add_argument(
    "--name", help="Name for experiment",
)
parser.add_argument(
    "--log_dir", help="Root for the directory where logs are saved", default=".",
)
parser.add_argument(
    "--data-dir", help="Path to data directory",
)
parser.add_argument("--data-variant", help="Data variant")
parser.add_argument(
    "--n_splits", help="Number of splits to train on", type=int, default=20
)
parser.add_argument(
    "--training", help="Type of training eg. likelihood.", default="likelihood"
)
parser.add_argument("--batch_size", help="Batch size", type=int, default=256)
parser.add_argument(
    "--n_updates", help="Number of updates for training", type=int, default=20000
)
parser.add_argument("--log_every", help="log_every", type=int, default=1)
parser.add_argument(
    "--flush-every", help="Flush logger to disk period", type=int, default=10
)
parser.add_argument(
    "--hidden_dims", help="Hidden dimensions for NN", nargs="+", type=int, default=[50],
)
parser.add_argument(
    "--loss-weight", help="Weighting param for loss (beta)", type=float, default=0.0
)


def train_on_split(
    data_dir,
    data_variant,
    split_idx,
    logger,
    dummy_logger,
    training_method,
    n_updates,
    batch_size,
    hidden_dims,
    grid_setting_names,
    grid_settings,
    log_every,
    loss_weight=0.0,
    train_split=0.8,
    device="cpu",
):
    def train_model(
        n_updates, train_val_split, lr=3e-4, activation="relu", n_epochs=None
    ):
        full_split = train_val_split == 1.0
        train_dataset, eval_data, inp_dim, outp_dim = data.load_dataset(
            dataset_name="uci",
            data_variant=data_variant,
            train_split=train_val_split,
            test_split_idx=split_idx,
            data_dir=data_dir,
            load_test_set=full_split,
        )
        if full_split:
            eval_dataset = None
        else:
            eval_dataset, _, _ = eval_data

        n_train = len(train_dataset)
        batch_size_ = min(n_train, batch_size)

        if n_epochs is None:
            n_updates_per_epochs = int(math.floor(n_train / batch_size_))
            n_epochs = int(math.ceil(n_updates / n_updates_per_epochs))

        cmds = [
            f"--training={training_method}",
            f"--lr={lr}",
            f"--optimizer_name=adam",
            f"--n_epochs={n_epochs}",
            f"--batch_size={batch_size_}",
            f"--log_every={log_every}",
            f"--hidden_activation={activation}",
            f"--loss-weight={loss_weight}",
            f"--early-stop-iters=50",
            f"--device={device}",
        ]
        if training_method == "mse":
            cmds.append("--early-stop-metric=eval_mse")
        else:
            cmds.append("--early-stop-metric=eval_likelihood")
        opts = opts_parser.parse_args(cmds)

        head, trainer = tutils.get_head_and_trainer(opts, hooks=[])
        model = MLP(
            inp_dim=inp_dim,
            outp_dim=outp_dim,
            hidden_dims=hidden_dims,
            hidden_activation=nnutils.get_activation(opts),
            weight_init=nnutils.get_weight_init(opts),
            bias_init=torch.nn.init.zeros_,
            outp_layer=head,
        )
        train_info = trainer.train(model, train_dataset, eval_dataset, dummy_logger)

        return trainer, model, train_info

    best_metric = np.inf
    best_conf = None
    best_epochs = 0

    for settings in itertools.product(*grid_settings):
        conf = {grid_setting_names[idx]: value for idx, value in enumerate(settings)}
        logger.info(f"Running settings {conf}")

        trainer, _, train_info = train_model(
            n_updates, train_val_split=train_split, **conf
        )

        if trainer.early_stop_best < best_metric:
            best_metric = trainer.early_stop_best
            best_conf = conf
            if "val_nlls" in train_info:
                best_epochs = np.argmin(train_info["val_nlls"]) + 1
                logger.info(
                    f"Best validation NLL changed to {best_metric:.5f} in {best_epochs}"
                )
            else:
                best_epochs = np.argmin(train_info["val_mses"]) + 1
                logger.info(
                    f"Best validation RMSE changed to {np.sqrt(best_metric):.5f} in {best_epochs}"
                )

    logger.info(f"Training final model for split with {best_conf}")
    trainer, eval_model, _ = train_model(
        n_updates=None, train_val_split=1.0, n_epochs=best_epochs, **best_conf
    )

    _, test_data, _, _ = data.load_dataset(
        dataset_name="uci",
        data_variant=data_variant,
        train_split=1.0,
        test_split_idx=split_idx,
        data_dir=data_dir,
        load_test_set=True,
    )
    test_dataset = test_data[0]
    test_x, test_y = trainer.val_dataset_to_torch(test_dataset)

    test_mse = trainer.eval_mse(eval_model, test_x, test_y).cpu()
    if hasattr(trainer, "eval_likelihood"):
        test_nll = trainer.eval_likelihood(eval_model, test_x, test_y).cpu()
    else:
        test_nll = np.nan
    logger.info(f"Test RMSE: {np.sqrt(test_mse):.5f}, NLL: {test_nll:.5f}")

    return test_mse, test_nll


def main(opts):
    if opts.seed is None:
        opts.seed = random.randint(0, 1e8)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    dirname = time.strftime(f"%y-%m-%d-%H%M%S", time.gmtime(time.time()))
    if opts.name is not None:
        dirname += "-" + opts.name
    log_path = os.path.join(opts.log_dir, dirname)
    opts.log_dir = log_path
    logger = Logger(
        root_dir=opts.log_dir, dry=opts.dry, quiet=False, flush_every=opts.flush_every
    )
    logger.info(opts)
    logger.info(f"Random seed is {opts.seed}")
    logger.log_config(opts)

    grid_setting_names = ["lr"]
    grid_settings = [[1e-4, 3e-4, 7e-4, 1e-3, 3e-3, 7e-3]]

    n_splits = min(opts.n_splits, data.uci_datasets.N_SPLITS[opts.data_variant])

    test_mses, test_nlls = [], []
    for split_idx in range(n_splits):
        logger.info(f"############  Running split {split_idx}  ############")
        dummy_logger = Logger(root_dir=opts.log_dir, dry=True, quiet=True)

        mse, nll = train_on_split(
            opts.data_dir,
            opts.data_variant,
            split_idx,
            logger,
            dummy_logger,
            opts.training,
            opts.n_updates,
            opts.batch_size,
            opts.hidden_dims,
            grid_setting_names,
            grid_settings,
            opts.log_every,
            opts.loss_weight,
            device=opts.device,
        )
        test_mses.append(mse)
        test_nlls.append(nll)

    logger.info(
        f"Final test RMSE: {np.sqrt(np.mean(test_mses)):.5f}, NLL: {np.mean(test_nlls):.5f}"
    )

    training_info = {"test_mses": test_mses, "test_nlls": test_nlls}
    logger.log_training_as_dataframe(training_info)

    try:
        logger.log_training(training_info)
    except Exception as e:
        logger.info(f"Logging training failed: {e}")

    return training_info


if __name__ == "__main__":
    opts = parser.parse_args()
    metrics = main(opts)
