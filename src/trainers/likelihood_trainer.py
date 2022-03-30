from collections import defaultdict

import numpy as np
import torch
from datetime import datetime

from src.models.utils import (
    gaussian_log_likelihood_loss,
    gaussian_beta_log_likelihood_loss,
)
from .base_trainer import BaseTrainer
from .hooks import HookPhase


class LikelihoodTrainer(BaseTrainer):
    def __init__(
        self,
        collect_likelihoods=False,
        optimizer_name="adam",
        mse_warmup=0,
        lr=1e-4,
        n_epochs=100,
        betas=[0.9, 0.999],
        batch_size=512,
        log_every=5,
        checkpoint=20,
        device="cpu",
        hooks=None,
        momentum=0,
        rms_alpha=0.99,
        detach_variance=False,
        dataloader_drop_last=True,
        nll_weight_beta=0.0,
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
            rms_alpha=rms_alpha,
            early_stop_metric=early_stop_metric,
            early_stop_iters=early_stop_iters,
            track_best_metrics=track_best_metrics,
            log_intermediate_results=log_intermediate_results,
        )

        if nll_weight_beta > 0.0:
            self.loss_fn = lambda pred, target: (
                gaussian_beta_log_likelihood_loss(pred, target, nll_weight_beta)
            )
        else:
            self.loss_fn = gaussian_log_likelihood_loss
        self.nll_weight_beta = nll_weight_beta
        self.mse_warmup = mse_warmup
        self.detach_variance = detach_variance
        self.collect_likelihoods = collect_likelihoods
        self.dataloader_drop_last = dataloader_drop_last

    def step(self, model, inp, target, epoch):
        if self.model_regularizer is not None:
            pred = model(inp, with_regularization=True)
        else:
            pred = model(inp)
        mean, variance = pred[:2]

        if epoch < self.mse_warmup:
            variance = torch.ones_like(mean)
        if self.detach_variance:
            variance = variance.detach()

        loss = self.loss_fn((mean, variance), target)

        if self.model_regularizer is not None:
            reg_loss = self.model_regularizer(pred)
            loss += reg_loss
            return mean, variance, loss, reg_loss
        else:
            return mean, variance, loss

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
        optimizer = self._get_optimizer(model.parameters())
        should_early_stop = False
        self.execute_hooks_for_phase(HookPhase.INIT, model, optimizer, dataloader)

        for epoch in range(self.n_epochs):
            self.execute_hooks_for_phase(
                HookPhase.EPOCH_START, model, dataloader, epoch
            )
            model.train()

            epoch_losses, epoch_reg_losses = [], []
            epoch_mses, epoch_stds, epoch_likelihoods = [], [], []
            epoch_hlikelihoods, epoch_hmses = [], []
            epoch_elikelihoods, epoch_emses = [], []
            for idx, batch in enumerate(dataloader):
                global_step = epoch * len(dataloader) + idx

                inp = batch[0].float().to(self.device)
                target = batch[1].float().to(self.device)

                if self.collect_likelihoods and len(batch) == 3:
                    label = batch[2]

                self.execute_hooks_for_phase(
                    HookPhase.MODEL_FORWARD, model, inp, target
                )

                if self.model_regularizer is None:
                    mean, variance, loss = self.step(model, inp, target, epoch)
                else:
                    mean, variance, loss, reg_loss = self.step(
                        model, inp, target, epoch
                    )

                pred = mean, variance
                self.execute_hooks_for_phase(
                    HookPhase.MODEL_FORWARD_END, model, inp, pred, loss, target
                )
                mean_loss = loss.mean()

                with torch.no_grad():
                    # Collect basic training data
                    mean_, variance_ = self.denorm(mean, variance)
                    target_ = self.denorm(target)
                    mse = (target_ - mean_) ** 2
                    nll = gaussian_log_likelihood_loss((mean_, variance_), target_)
                    epoch_losses.append(mean_loss.detach().cpu().numpy())
                    epoch_mses.append(mse.mean(dim=-1).detach().cpu().numpy())
                    epoch_stds.append(
                        variance_.mean(dim=-1).sqrt().detach().cpu().numpy()
                    )
                    epoch_likelihoods.append(nll.mean(dim=-1).detach().cpu().numpy())
                    if self.model_regularizer is not None:
                        epoch_reg_losses.append(reg_loss.detach().cpu().numpy())

                    if self.collect_likelihoods:
                        hard = (label == 1).squeeze()
                        easy = (label == 0).squeeze()

                        # Collect MSE for hard and easy samples
                        epoch_hmses.append(mse[hard].mean(dim=0).detach().cpu().numpy())
                        epoch_emses.append(mse[easy].mean(dim=0).detach().cpu().numpy())

                        # Collect likelihood values for hard and easy samples
                        epoch_hlikelihoods.append(
                            loss[hard].mean(dim=0).detach().cpu().numpy()
                        )

                        epoch_elikelihoods.append(
                            loss[easy].mean(dim=0).detach().cpu().numpy()
                        )

                optimizer.zero_grad()
                self.backward(
                    global_step, self.loss_fn(pred, target), mean_loss, optimizer
                )
                self.execute_hooks_for_phase(HookPhase.BACKWARD, model)

                optimizer.step()
                self.execute_hooks_for_phase(HookPhase.OPT_STEP, model)

            self.execute_hooks_for_phase(HookPhase.EPOCH_END, model, dataloader, epoch)

            info_dict["losses"].append(np.mean(epoch_losses))
            if len(epoch_reg_losses) > 0:
                info_dict["reg_losses"].append(np.mean(epoch_reg_losses))
            info_dict["mses"].append(np.mean(epoch_mses))
            info_dict["stds"].append(np.mean(epoch_stds))
            info_dict["nlls"].append(np.mean(epoch_likelihoods))

            if self.collect_likelihoods:
                info_dict["hlikelihoods"].append(np.mean(epoch_hlikelihoods))
                info_dict["elikelihoods"].append(np.mean(epoch_elikelihoods))
                info_dict["hmses"].append(np.mean(epoch_hmses))
                info_dict["emses"].append(np.mean(epoch_emses))

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
                    if self.nll_weight_beta > 0.0:
                        eval_loss = self.eval_loss(
                            model, eval_x, eval_y, pred=pred
                        ).cpu()
                    else:
                        eval_loss = eval_likelihood.cpu()
                    info_dict["val_epochs"].append(epoch)
                    info_dict["val_mses"].append(eval_mse.cpu().numpy())
                    info_dict["val_nlls"].append(eval_likelihood.cpu().numpy())
                    info_dict["val_losses"].append(eval_loss.cpu().numpy())
                    should_early_stop = self.early_stopping(
                        model,
                        logger,
                        epoch,
                        eval_mse=info_dict["val_mses"][-1],
                        eval_likelihood=info_dict["val_nlls"][-1],
                        eval_loss=info_dict["val_losses"][-1],
                    )

                log_str = (
                    f"Epoch {epoch}: {info_dict['losses'][-1]:.4f}, rmse: {np.sqrt(info_dict['mses'][-1]):.4f}, "
                    f"std: {info_dict['stds'][-1]:.4f}"
                )
                if "reg_losses" in info_dict:
                    log_str += f", reg_loss: {info_dict['reg_losses'][-1]:.4f}"
                if eval_x is not None:
                    log_str += f", eval rmse: {np.sqrt(eval_mse):.4f}, eval NLL: {eval_likelihood:.4f}"
                    if self.nll_weight_beta > 0.0:
                        log_str += f", eval loss: {eval_loss:.4f}"
                logger.info(log_str)

                self.log(logger, epoch)

            if self.checkpoint > 0 and (
                epoch % self.checkpoint == 0
                or epoch == self.n_epochs - 1
                or epoch == self.mse_warmup - 1
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
        pred = model(torch.as_tensor(xs, dtype=torch.float))

        mean, variance = self.denorm(*pred)

        return mean.cpu().numpy(), variance.cpu().numpy()

    @torch.no_grad()
    def eval_likelihood(self, model, inp, target, pred=None):
        inp = inp.to(self.device)
        target = target.to(self.device)
        if pred is None:
            if model.training:
                model.eval()
            pred = model(inp)
        nll = gaussian_log_likelihood_loss(self.denorm(*pred), target)

        return nll.mean()

    @torch.no_grad()
    def eval_loss(self, model, inp, target, pred=None):
        inp = inp.to(self.device)
        target = target.to(self.device)
        if pred is None:
            if model.training:
                model.eval()
            pred = model(inp)
        loss = self.loss_fn(self.denorm(*pred), target)

        return loss.mean()
