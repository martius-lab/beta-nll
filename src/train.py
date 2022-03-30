import os
import random
import time
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

from src import data
from src.models import get_model
from src.trainers import utils as tutils
from src.trainers.hooks import HookPhase
from src.utils.viz import Plotter
from src.utils.log import Logger

parser = ArgumentParser()
parser.add_argument(
    "-n", "--dry", help="Set if running only for sanity check", action="store_true"
)
parser.add_argument("--seed", help="Random seed", type=int)
parser.add_argument(
    "--name", help="Name for experiment",
)
parser.add_argument("--device", help="Device name for training", default="cpu")
parser.add_argument("--q", help="Quiet mode (no prints)", default=False)
parser.add_argument(
    "--data-dir", help="Path to data directory",
)
parser.add_argument(
    "--dataset",
    help="Which data should be generated for the training purposes",
    default="10",
)
parser.add_argument("--data_variant", help="Data variant, if any")
parser.add_argument(
    "--standardize-inputs",
    help="Standardize inputs",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--use_diff_as_target",
    help="Use difference between positions as target",
    action="store_true",
    default=True,
)
parser.add_argument(
    "--train-split",
    help="Portion of data that should be used for training. (Rest will be val/test set); between [0, 1]",
    default=None,
    type=float,
)
parser.add_argument(
    "--test-split",
    help="Portion of data that should be used for testing. (Rest will be validation set); between [0, 1]",
    default=None,
    type=float,
)
parser.add_argument(
    "--data_only_random",
    help="Use only random actions in dataset",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--test-split-idx", help="Index of the test split to use", default=0, type=int,
)
parser.add_argument(
    "--data-noise-level", help="Level of artificial noise to add to data", type=float,
)
parser.add_argument(
    "--collect_likelihoods",
    help="Set if collecting of likelihoods during training is needed",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--hidden_activation", help="Hidden activation function name", default="tanh"
)
parser.add_argument(
    "--weight-init", help="Weight initialization", default="xavier_uniform"
)
parser.add_argument(
    "--batchnorm_first",
    help="Normalize input with batchnorm",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--training", help="Type of training eg. likelihood|mse.", default="mse"
)
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-2)
parser.add_argument(
    "--momentum", help="Momentum param for SGD or RMSprop", type=float, default=0
)
parser.add_argument("--batch_size", help="Batch size", type=int, default=128)
parser.add_argument(
    "--n_epochs", help="Number of epochs for training", type=int, default=50
)
parser.add_argument("--log_every", help="log_every", type=int, default=1000)
parser.add_argument(
    "--checkpoint", help="Checkpoint the model", type=int, default=10000
)
parser.add_argument("--mse_warmup", help="Warmup epochs for MSE", type=int, default=0)
parser.add_argument(
    "--detach_variance",
    help="Detach variance in likelihood training",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--hidden_dims",
    help="Hidden dimensions for NN",
    nargs="+",
    type=int,
    default=[256, 256],
)
parser.add_argument(
    "--model-type", help="NN to use", default="MLP",
)
parser.add_argument(
    "--latent-dims", help="Latent dimensions for VAE", default=10, type=int
)
parser.add_argument(
    "--spectral-norm",
    help="Whether to use spectral norm in NN",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--dataloader-keep-last",
    help="Whether to keep last uneven batch in dataloader",
    action="store_true",
)
parser.add_argument(
    "--optimizer_name", help="Optimizer name", default="adam",
)
parser.add_argument(
    "--betas",
    help="Betas parameters for Adam optimizer",
    nargs=2,
    type=float,
    default=[0.9, 0.999],
)
parser.add_argument(
    "--rms-alpha",
    help="Alpha parameter for RMSprop optimizer",
    type=float,
    default=0.99,
)
parser.add_argument("--loss-weight", help="Weighting param for loss (beta)", type=float)
parser.add_argument(
    "--log_dir", help="Root for the directory where logs are saved", default=".",
)
parser.add_argument(
    "--log-intermediate-results",
    help="Save metrics already during training",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--no-plotting",
    help="Set true to skip plots at end of training",
    action="store_true",
    default=False,
)
parser.add_argument("--load_checkpoint", help="Checkpoint to start from")
parser.add_argument(
    "--eval-test",
    help="Eval test set after training",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--hooks-every", help="Frequency to run hooks with, in epochs", nargs="+", type=int
)
parser.add_argument(
    "--stop-hooks-after",
    help="Epoch after which to stop evaluating hooks",
    nargs="+",
    type=int,
)
parser.add_argument(
    "--num-hook-points", help="Number of points to use for hooks", default=200, type=int
)
parser.add_argument(
    "--eval-hook", help="Add eval hook", action="store_true", default=False,
)
parser.add_argument(
    "--jac-var-hook",
    help="Add jacobian variance hook",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--feature-layer", help="Index of layer to use for feature analysis", type=int
)
parser.add_argument(
    "--granularities", help="Granularities to use for hooks", nargs="+", type=float
)
parser.add_argument(
    "--vae-vis-hook",
    help="Add VAE visualization hook",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--test-posterior-metrics-hook",
    help="Add VAE visualization hook",
    action="store_true",
    default=False,
)
parser.add_argument("--early-stop-metric", help="Metric to use for early stopping")
parser.add_argument(
    "--early-stop-iters",
    help="Stop after this many iterations with no improvement",
    type=int,
)
parser.add_argument(
    "--track-best-metrics", help="Metrics to save best model for", nargs="+"
)


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
    logger = Logger(root_dir=opts.log_dir, dry=opts.dry, quiet=opts.q,)
    logger.info(opts)
    logger.info(f"Random seed is {opts.seed}")
    logger.log_config(opts)

    train_dataset, eval_data, inp_dim, outp_dim = data.load_dataset(
        opts.dataset,
        opts.data_variant,
        opts.train_split,
        opts.test_split,
        opts.data_only_random,
        opts.use_diff_as_target,
        opts.standardize_inputs,
        opts.test_split_idx,
        opts.data_noise_level,
        opts.data_dir,
    )
    eval_dataset, eval_x, eval_y = eval_data

    collect_likelihoods = data.should_collect_likelihoods(opts.dataset)
    if collect_likelihoods is not None:
        opts.collect_likelihoods = collect_likelihoods

    hooks = tutils.get_hooks(opts, train_dataset, eval_dataset)

    head, trainer = tutils.get_head_and_trainer(opts, hooks)
    model = get_model(inp_dim, outp_dim, head, opts)

    if opts.load_checkpoint is not None:
        logger.info(f"Loading model checkpoint from {opts.load_checkpoint}")
        state = torch.load(opts.load_checkpoint)
        model.load_state_dict(state)

    viz = Plotter(opts.log_dir, dry=opts.dry)

    training_info = trainer.train(model, train_dataset, eval_dataset, logger)
    logger.save_hooks(hooks)

    if opts.eval_test:
        _, test_data, _, _ = data.load_dataset(
            opts.dataset,
            opts.data_variant,
            opts.train_split,
            opts.test_split,
            opts.data_only_random,
            opts.use_diff_as_target,
            opts.standardize_inputs,
            opts.test_split_idx,
            opts.data_noise_level,
            opts.data_dir,
            load_test_set=True,
        )
        test_dataset = test_data[0]
        test_x, test_y = trainer.val_dataset_to_torch(test_dataset)

        eval_models = {"last": model}
        if opts.early_stop_metric is not None and trainer.early_stop_model is not None:
            eval_models["early_stop"] = trainer.early_stop_model
        if opts.track_best_metrics is not None:
            eval_models.update(
                {f"best_{m}": model for m, model in trainer.track_best_models.items()}
            )

        for name, eval_model in eval_models.items():
            test_mse = trainer.eval_mse(eval_model, test_x, test_y).cpu()
            training_info[f"test_{name}_mse"] = [test_mse.numpy()]

            if hasattr(trainer, "eval_likelihood"):
                test_nll = trainer.eval_likelihood(eval_model, test_x, test_y).cpu()
                training_info[f"test_{name}_nll"] = [test_nll.numpy()]
                logger.info(
                    f"Model {name}: Test RMSE: {np.sqrt(test_mse):.4f}, test NLL: {test_nll:.4f}"
                )
            else:
                logger.info(f"Model {name}: Test RMSE: {np.sqrt(test_mse):.4f}")
            if hasattr(trainer, "eval_loss"):
                test_loss = trainer.eval_loss(eval_model, test_x, test_y).cpu()
                training_info[f"test_{name}_loss"] = [test_loss.numpy()]

            trainer.execute_hooks_for_phase(
                HookPhase.TEST, eval_model, name, test_x, test_y, training_info, logger,
            )

    logger.log_training_as_dataframe(training_info)

    if not opts.no_plotting:
        try:
            viz.plot_training(training_info)
        except Exception as e:
            logger.info(f"Plotting training failed: {e}")

    y_mean, y_var = trainer.eval_model(model, eval_x)
    eval_rmse = torch.sqrt(
        F.mse_loss(torch.from_numpy(y_mean), torch.from_numpy(eval_y))
    )

    training_info["eval_rmse"] = [eval_rmse]
    training_info["y_var"] = [np.mean(y_var)]
    try:
        logger.log_training(training_info)
    except Exception as e:
        logger.info(f"Logging training failed: {e}")

    if not opts.no_plotting and inp_dim <= 2 and outp_dim == 1:
        try:
            viz.plot_data(
                train_dataset.X,
                train_dataset.pts,
                eval_x,
                y_mean,
                y_var,
                ground_truth=train_dataset.Y,
            )
        except Exception as e:
            logger.info(f"Plotting data failed: {e}")

    return tutils.filter_dictionary(training_info)


if __name__ == "__main__":
    opts = parser.parse_args()
    metrics = main(opts)
