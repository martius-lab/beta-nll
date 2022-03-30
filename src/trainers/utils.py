import numpy as np
from torch import nn

from src.trainers import hooks as hooks_
from .likelihood_trainer import LikelihoodTrainer
from .moment_matching_trainer import MomentMatchingTrainer
from .mse_trainer import MSETrainer
from .student_t_trainer import StudentTTrainer
from .variational_variance_trainer import VariationalVarianceTrainer
from src.models import variational_variance
from src.models.networks import (
    GaussianLikelihoodHead,
    NormalGammaHead,
    GaussianLikelihoodFixedVarianceHead,
)


def filter_dictionary(info_dict):
    non_metric = [
        "hgrad_norm",
        "egrad_norm",
        "ehgrad_angle",
        "ehgrad_norm_sim",
        "ehgrad_conflict",
        "wanalysis",
    ]

    return dict(filter(lambda x: x[0] not in non_metric, info_dict.items()))


def get_head_and_trainer(opts, hooks=None):
    if "likelihood" in opts.training:
        if "fixed_unit_var" in opts.training:
            head = lambda inp, outp: GaussianLikelihoodFixedVarianceHead(
                inp, outp, fixed_var=1
            )
        else:
            head = lambda inp, outp: GaussianLikelihoodHead(
                inp, outp, initial_var=1, max_var=100
            )

        nll_weight_beta = 0.0 if opts.loss_weight is None else opts.loss_weight

        trainer = LikelihoodTrainer(
            collect_likelihoods=opts.collect_likelihoods,
            optimizer_name=opts.optimizer_name,
            mse_warmup=opts.mse_warmup,
            lr=opts.lr,
            n_epochs=opts.n_epochs,
            betas=opts.betas,
            batch_size=opts.batch_size,
            log_every=opts.log_every,
            checkpoint=opts.checkpoint,
            device=opts.device,
            hooks=hooks,
            momentum=opts.momentum,
            rms_alpha=opts.rms_alpha,
            detach_variance=opts.detach_variance,
            dataloader_drop_last=not opts.dataloader_keep_last,
            nll_weight_beta=nll_weight_beta,
            early_stop_metric=opts.early_stop_metric,
            early_stop_iters=opts.early_stop_iters,
            track_best_metrics=opts.track_best_metrics,
            log_intermediate_results=opts.log_intermediate_results,
        )
    elif opts.training == "mse":
        head = nn.Linear

        trainer = MSETrainer(
            optimizer_name=opts.optimizer_name,
            lr=opts.lr,
            n_epochs=opts.n_epochs,
            betas=opts.betas,
            batch_size=opts.batch_size,
            log_every=opts.log_every,
            checkpoint=opts.checkpoint,
            device=opts.device,
            hooks=hooks,
            momentum=opts.momentum,
            dataloader_drop_last=not opts.dataloader_keep_last,
            early_stop_metric=opts.early_stop_metric,
            early_stop_iters=opts.early_stop_iters,
            track_best_metrics=opts.track_best_metrics,
            log_intermediate_results=opts.log_intermediate_results,
        )
    elif opts.training.startswith("moment_matching"):
        head = lambda inp, outp: GaussianLikelihoodHead(
            inp, outp, initial_var=1, max_var=100
        )

        gamma_loss = "gamma" in opts.training

        trainer = MomentMatchingTrainer(
            collect_likelihoods=opts.collect_likelihoods,
            optimizer_name=opts.optimizer_name,
            lr=opts.lr,
            n_epochs=opts.n_epochs,
            betas=opts.betas,
            batch_size=opts.batch_size,
            log_every=opts.log_every,
            checkpoint=opts.checkpoint,
            device=opts.device,
            hooks=hooks,
            momentum=opts.momentum,
            gamma_loss=gamma_loss,
            dataloader_drop_last=not opts.dataloader_keep_last,
            early_stop_metric=opts.early_stop_metric,
            early_stop_iters=opts.early_stop_iters,
            track_best_metrics=opts.track_best_metrics,
            log_intermediate_results=opts.log_intermediate_results,
        )
    elif opts.training == "student_t":
        head = lambda inp, outp: NormalGammaHead(inp, outp)

        trainer = StudentTTrainer(
            collect_likelihoods=opts.collect_likelihoods,
            optimizer_name=opts.optimizer_name,
            lr=opts.lr,
            n_epochs=opts.n_epochs,
            betas=opts.betas,
            batch_size=opts.batch_size,
            log_every=opts.log_every,
            checkpoint=opts.checkpoint,
            device=opts.device,
            hooks=hooks,
            momentum=opts.momentum,
            dataloader_drop_last=not opts.dataloader_keep_last,
            early_stop_metric=opts.early_stop_metric,
            early_stop_iters=opts.early_stop_iters,
            track_best_metrics=opts.track_best_metrics,
            log_intermediate_results=opts.log_intermediate_results,
        )
    elif "vari_var" in opts.training:
        n_mc_samples = 20
        n_components = 100
        if "vbem" in opts.training:
            if "star" not in opts.training:
                n_components = 144

            def head(inp_dim, outp_dim):
                return variational_variance.NormalGammaPiHead(
                    inp_dim, outp_dim, n_components=n_components
                )

            def reg_init_fn(model, inp_dim, outp_dim, batch):
                if "star" in opts.training:
                    alphas, betas = variational_variance.get_vbem_star_standard_init(
                        outp_dim, n_components
                    )
                else:
                    alphas, betas = variational_variance.get_vbem_standard_init(
                        outp_dim
                    )
                return variational_variance.VBEMRegularizer(
                    alphas,
                    betas,
                    n_mc_samples=n_mc_samples,
                    prior_trainable="star" in opts.training,
                )

        elif "xvamp" in opts.training:

            def head(inp_dim, outp_dim):
                return variational_variance.NormalGammaPiHead(
                    inp_dim, outp_dim, n_components=n_components
                )

            def reg_init_fn(model, inp_dim, outp_dim, batch):
                inputs = batch[0].float()
                if len(inputs) < n_components:
                    raise ValueError(
                        "Not enough samples for xVAMP regularizer in batch"
                    )
                return variational_variance.xVAMPRegularizer(
                    model,
                    inputs[:n_components],
                    n_mc_samples=n_mc_samples,
                    prior_trainable="star" in opts.training,
                )

        else:
            raise ValueError("Unknown regularizer")

        trainer = VariationalVarianceTrainer(
            reg_init_fn,
            collect_likelihoods=opts.collect_likelihoods,
            optimizer_name=opts.optimizer_name,
            lr=opts.lr,
            n_epochs=opts.n_epochs,
            betas=opts.betas,
            batch_size=opts.batch_size,
            log_every=opts.log_every,
            checkpoint=opts.checkpoint,
            device=opts.device,
            hooks=hooks,
            momentum=opts.momentum,
            dataloader_drop_last=not opts.dataloader_keep_last,
            early_stop_metric=opts.early_stop_metric,
            early_stop_iters=opts.early_stop_iters,
            track_best_metrics=opts.track_best_metrics,
            log_intermediate_results=opts.log_intermediate_results,
        )
    else:
        raise ValueError(f"Unknown training method {opts.training}.")

    return head, trainer


def get_hooks(opts, train_dataset, eval_dataset):
    hooks = []
    if (
        not opts.eval_hook
        and not opts.jac_var_hook
        and not opts.vae_vis_hook
        and not opts.test_posterior_metrics_hook
    ):
        return hooks

    run_every = opts.hooks_every if opts.hooks_every else [opts.log_every]
    if not isinstance(run_every, (list, tuple)):
        run_every = [run_every]

    if opts.eval_hook or opts.jac_var_hook:
        example = train_dataset[0][0]
        if example.shape[0] == 1:  # Toy dataset
            inputs = [x[0] for x in train_dataset]
            eval_points = np.linspace(
                np.min(inputs), np.max(inputs), opts.num_hook_points
            )[:, None]
            targets = None
        else:
            rng = np.random.RandomState(44)
            idxs = rng.permutation(len(train_dataset))[: opts.num_hook_points]
            eval_points = train_dataset[idxs][0]
            targets = train_dataset[idxs][1]

        assert eval_points.ndim == 2

    if opts.eval_hook:
        hooks.append(
            hooks_.EvalHook(
                eval_points=eval_points,
                eval_every=run_every,
                last_epoch=opts.n_epochs,
                stop_after=opts.stop_hooks_after,
                targets=targets,
            )
        )
    if opts.jac_var_hook:
        assert opts.feature_layer is not None
        assert opts.granularities is not None
        hook = hooks_.JacobianVarianceHook(
            eval_points=eval_points,
            granularities=opts.granularities,
            target_layer=opts.feature_layer,
            eval_every=run_every,
            last_epoch=opts.n_epochs,
            stop_after=opts.stop_hooks_after,
        )
        hooks.append(hook)
    if opts.vae_vis_hook:
        hook = hooks_.VAEVisualizationHook(
            eval_every=run_every,
            last_epoch=opts.n_epochs,
            image_size=train_dataset.shape,
        )
        hooks.append(hook)
    if opts.test_posterior_metrics_hook:
        hooks.append(hooks_.TestPosteriorPredictiveMetricsHook())

    return hooks
