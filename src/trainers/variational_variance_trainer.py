import itertools
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from src.models.utils import student_t_nll
from src.models.variational_variance import exp_gaussian_nll_under_gamma_prec
from .base_trainer import BaseTrainer
from .hooks import HookPhase


class VariationalVarianceTrainer(BaseTrainer):
    def __init__(
        self,
        regularizer_init_fn,
        collect_likelihoods=False,
        optimizer_name="adam",
        lr=1e-4,
        n_epochs=100,
        betas=[0.9, 0.999],
        batch_size=512,
        log_every=5,
        checkpoint=20,
        device="cuda",
        hooks=None,
        momentum=0,
        dataloader_drop_last=True,
        early_stop_metric=None,
        early_stop_iters=None,
        track_best_metrics=None,
        log_intermediate_results=False,
    ):
        super().__init__(
            optimizer_name,
            lr,
            n_epochs,
            betas,
            batch_size,
            log_every,
            checkpoint,
            device,
            hooks,
            momentum=momentum,
            early_stop_metric=early_stop_metric,
            early_stop_iters=early_stop_iters,
            track_best_metrics=track_best_metrics,
            log_intermediate_results=log_intermediate_results,
        )

        self.loss_fn = lambda pred, target: exp_gaussian_nll_under_gamma_prec(
            *pred[:3], target
        )
        self.regularizer_init_fn = regularizer_init_fn
        self.regularizer = None
        self.dataloader_drop_last = dataloader_drop_last

    def train(self, model, dataset, val_dataset, logger):
        logger.info(f"Training started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

        if hasattr(dataset, "target_mean") and hasattr(dataset, "target_std"):
            self.set_target_normalizer(dataset.target_mean, dataset.target_std)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=self.dataloader_drop_last,
        )
        if self.log_every >= 0:
            logger.info(f"Num samples per epoch: {len(dataloader) * self.batch_size}")

        info_dict = defaultdict(list)

        if val_dataset is not None:
            eval_x, eval_y = self.val_dataset_to_torch(val_dataset)
        else:
            eval_x, eval_y = None, None

        model = self.init_model(model)

        if self.regularizer is None:
            batch = next(iter(dataloader))
            inp_dim = batch[0].shape[-1]
            outp_dim = batch[1].shape[-1]
            regularizer = self.regularizer_init_fn(model, inp_dim, outp_dim, batch)
            self.regularizer = regularizer.to(self.device)

        optimizer = self._get_optimizer(
            itertools.chain(model.parameters(), self.regularizer.parameters())
        )
        should_early_stop = False
        self.execute_hooks_for_phase(HookPhase.INIT, model, optimizer, dataloader)

        for epoch in range(self.n_epochs):
            self.execute_hooks_for_phase(
                HookPhase.EPOCH_START, model, dataloader, epoch
            )
            model.train()

            epoch_losses, epoch_reg_losses, epoch_var_reg_losses = [], [], []
            epoch_mses, epoch_stds, epoch_likelihoods, epoch_dof = [], [], [], []
            for idx, batch in enumerate(dataloader):
                global_step = epoch * len(dataloader) + idx

                inp = batch[0].float().to(self.device)
                target = batch[1].float().to(self.device)

                self.execute_hooks_for_phase(
                    HookPhase.MODEL_FORWARD, model, inp, target
                )

                if self.model_regularizer is not None:
                    pred = model(inp, with_regularization=True)
                else:
                    pred = model(inp)

                mean, alpha, beta = pred[:3]

                loss = self.loss_fn(pred[:4], target)
                var_reg_loss = self.regularizer(pred[:4])

                if self.model_regularizer is not None:
                    reg_loss = self.model_regularizer(pred)
                    loss += reg_loss

                self.execute_hooks_for_phase(
                    HookPhase.MODEL_FORWARD_END, model, inp, pred, loss, target
                )
                mean_loss = (loss + var_reg_loss).mean()

                with torch.no_grad():
                    # Collect basic training data
                    dof = 2 * alpha
                    scale_squared = beta / alpha
                    mean_, scale_squared_ = self.denorm(mean, scale_squared)
                    target_ = self.denorm(target)
                    mse = (target_ - mean_) ** 2

                    nll = student_t_nll(dof, mean_, torch.sqrt(scale_squared_), target_)

                    # Assume that dof > 2 by NN parametrization
                    variance = scale_squared_ * dof / (dof - 2)

                    epoch_losses.append(mean_loss.detach().cpu().numpy())
                    epoch_var_reg_losses.append(var_reg_loss.detach().cpu().numpy())
                    epoch_mses.append(mse.mean(dim=-1).detach().cpu().numpy())
                    epoch_stds.append(
                        variance.mean(dim=-1).sqrt().detach().cpu().numpy()
                    )
                    epoch_likelihoods.append(nll.mean(dim=-1).detach().cpu().numpy())
                    epoch_dof.append(dof.mean(dim=-1).detach().cpu().numpy())
                    if self.model_regularizer is not None:
                        epoch_reg_losses.append(reg_loss.detach().cpu().numpy())

                optimizer.zero_grad()
                self.backward(global_step, None, mean_loss, optimizer)
                self.execute_hooks_for_phase(HookPhase.BACKWARD, model)

                optimizer.step()
                self.execute_hooks_for_phase(HookPhase.OPT_STEP, model)

            self.execute_hooks_for_phase(HookPhase.EPOCH_END, model, dataloader, epoch)

            info_dict["losses"].append(np.mean(epoch_losses))
            info_dict["var_reg_losses"].append(np.mean(epoch_var_reg_losses))
            if len(epoch_reg_losses) > 0:
                info_dict["reg_losses"].append(np.mean(epoch_reg_losses))
            info_dict["mses"].append(np.mean(epoch_mses))
            info_dict["stds"].append(np.mean(epoch_stds))
            info_dict["nlls"].append(np.mean(epoch_likelihoods))
            info_dict["dofs"].append(np.mean(epoch_dof))

            if self.log_every > 0 and (
                epoch % self.log_every == 0 or epoch == self.n_epochs - 1
            ):
                if eval_x is not None:
                    eval_x = eval_x.to(self.device)
                    with torch.no_grad():
                        if model.training:
                            model.eval()
                        pred = model(eval_x)
                    eval_mse = self.eval_mse(model, eval_x, eval_y, pred=pred).cpu()
                    eval_likelihood = self.eval_likelihood(
                        model, eval_x, eval_y, pred=pred
                    ).cpu()
                    eval_loss = eval_likelihood
                    info_dict["val_epochs"].append(epoch)
                    info_dict["val_mses"].append(eval_mse.numpy())
                    info_dict["val_nlls"].append(eval_likelihood.numpy())
                    info_dict["val_losses"].append(eval_loss.numpy())
                    should_early_stop = self.early_stopping(
                        model,
                        logger,
                        epoch,
                        eval_mse=info_dict["val_mses"][-1],
                        eval_likelihood=info_dict["val_nlls"][-1],
                        eval_loss=info_dict["val_losses"][-1],
                    )

                log_str = (
                    f"Epoch {epoch}: {info_dict['losses'][-1]:.4f}, "
                    f"var_reg_loss: {info_dict['var_reg_losses'][-1]:.4f}, "
                    f"rmse: {np.sqrt(info_dict['mses'][-1]):.4f}, "
                    f"std: {info_dict['stds'][-1]:.4f}, dof: {info_dict['dofs'][-1]:.2f}"
                )
                if "reg_losses" in info_dict:
                    log_str += f", reg_loss: {info_dict['reg_losses'][-1]:.4f}"
                if eval_x is not None:
                    log_str += f", eval rmse: {np.sqrt(eval_mse):.4f}, eval NLL: {eval_likelihood:.4f}"
                logger.info(log_str)

                self.log(logger, epoch)

            if self.checkpoint > 0 and (
                epoch % self.checkpoint == 0
                or epoch == self.n_epochs - 1
                or should_early_stop
            ):
                self.save_checkpoint(logger, model, epoch)

            if should_early_stop:
                logger.info(
                    (
                        f"Early stopping due to no improvement for {self._early_stop_no_improvements} "
                        f"eval iters for metric {self.early_stop_metric} at value {self.early_stop_best:.4f}"
                    )
                )
                break

        logger.info(
            f"Training finished: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        )
        self.execute_hooks_for_phase(HookPhase.END, model, info_dict)

        return info_dict

    @torch.no_grad()
    def eval_model(self, model, xs):
        if model.training:
            model.eval()
        xs = torch.from_numpy(xs).to(self.device)
        mean, alpha, beta = model(torch.as_tensor(xs, dtype=torch.float))[:3]

        dof = 2 * alpha
        scale_squared = beta / alpha
        mean, scale_squared = self.denorm(mean, scale_squared)
        variance = scale_squared * dof / (dof - 2)

        return mean.cpu().numpy(), variance.cpu().numpy()

    @torch.no_grad()
    def eval_likelihood(self, model, inp, target, pred=None):
        inp = inp.to(self.device)
        target = target.to(self.device)
        if pred is None:
            if model.training:
                model.eval()
            pred = model(inp)

        mean, alpha, beta = pred[:3]

        dof = 2 * alpha
        scale_squared = beta / alpha
        mean_, scale_squared_ = self.denorm(mean, scale_squared)

        nll = student_t_nll(dof, mean_, torch.sqrt(scale_squared_), target)

        return nll.mean()

    @torch.no_grad()
    def eval_loss(self, model, inp, target, pred=None):
        return self.eval_likelihood(model, inp, target, pred)
