from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from src.models.utils import gaussian_log_likelihood_loss
from .base_trainer import BaseTrainer
from .hooks import HookPhase


def sigma_loss_mse(pred_sigma, target_mse):
    return torch.mean((pred_sigma - target_mse.sqrt()) ** 2, axis=-1)


class MomentMatchingTrainer(BaseTrainer):
    def __init__(
        self,
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
        gamma_loss=False,
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
            momentum,
            early_stop_metric=early_stop_metric,
            early_stop_iters=early_stop_iters,
            track_best_metrics=track_best_metrics,
            log_intermediate_results=log_intermediate_results,
        )
        self.loss_fn = lambda pred, target: ((pred[0] - target) ** 2).mean(axis=-1)

        if gamma_loss:
            self.sigma_loss_fn = lambda pred, target: gaussian_log_likelihood_loss(
                (pred[0].detach(), pred[1]), target
            )
            self.learns_variance = True
        else:
            self.sigma_loss_fn = lambda pred, target: sigma_loss_mse(
                pred[1], (pred[0].detach() - target) ** 2
            )
            self.learns_variance = False

        self.collect_likelihoods = collect_likelihoods
        self.dataloader_drop_last = dataloader_drop_last

    def train(self, model, dataset, val_dataset, logger):
        logger.info(f"Training started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

        if hasattr(dataset, "target_mean") and hasattr(dataset, "target_std"):
            self.set_target_normalizer(dataset.target_mean, dataset.target_std)

        model = self.init_model(model)
        params = model.parameters()
        optimizer = self._get_optimizer(params)

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

        should_early_stop = False
        self.execute_hooks_for_phase(HookPhase.INIT, model, optimizer, dataloader)

        for epoch in range(self.n_epochs):
            self.execute_hooks_for_phase(
                HookPhase.EPOCH_START, model, dataloader, epoch
            )
            model.train()

            epoch_loss, epoch_reg_losses = [], []
            epoch_mu_loss, epoch_sigma_loss = [], []
            epoch_mses, epoch_stds, epoch_likelihoods = [], [], []
            epoch_hlikelihoods, epoch_hmses = [], []
            epoch_elikelihoods, epoch_emses = [], []
            for batch in dataloader:
                inp = batch[0].float().to(self.device)
                target = batch[1].float().to(self.device)

                if self.collect_likelihoods and len(batch) == 3:
                    label = batch[2]

                self.execute_hooks_for_phase(
                    HookPhase.MODEL_FORWARD, model, inp, target
                )

                if self.model_regularizer is not None:
                    pred = model(inp, with_regularization=True)
                else:
                    pred = model(inp)

                mean, std_or_var = pred[:2]
                if self.learns_variance:
                    variance = std_or_var
                    std = variance.sqrt()
                else:
                    std = std_or_var
                    variance = std ** 2

                mu_loss = self.loss_fn((mean, variance), target)
                sigma_loss = self.sigma_loss_fn((mean, std_or_var), target)

                loss = mu_loss + sigma_loss
                if self.model_regularizer is not None:
                    reg_loss = self.model_regularizer(pred)
                    loss += reg_loss

                self.execute_hooks_for_phase(
                    HookPhase.MODEL_FORWARD_END, model, inp, pred, loss, target
                )
                mean_loss = loss.mean()

                with torch.no_grad():
                    # Collect basic training data
                    mean_, variance_ = self.denorm(mean, variance)
                    target_ = self.denorm(target)
                    mse = (target_ - mean_) ** 2
                    epoch_mu_loss.append(mu_loss.mean().detach().cpu().numpy())
                    epoch_sigma_loss.append(sigma_loss.mean().detach().cpu().numpy())
                    epoch_loss.append(loss.detach().cpu().numpy())
                    epoch_mses.append(mse.mean(dim=-1).detach().cpu().numpy())
                    epoch_stds.append(
                        variance_.mean(dim=-1).sqrt().detach().cpu().numpy()
                    )
                    if self.model_regularizer is not None:
                        epoch_reg_losses.append(reg_loss.detach().cpu().numpy())

                    epoch_likelihoods.append(
                        gaussian_log_likelihood_loss((mean_, variance_), target_)
                        .mean(dim=0)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    if self.collect_likelihoods:
                        hard = label == 1
                        easy = label == 0

                        # Collect MSE for hard and easy samples
                        epoch_hmses.append(mse[hard].mean(dim=0).detach().cpu().numpy())
                        epoch_emses.append(mse[easy].mean(dim=0).detach().cpu().numpy())

                        # Collect likelihood values for hard and easy samples
                        epoch_hlikelihoods.append(
                            gaussian_log_likelihood_loss(
                                (mean_[hard], variance_[hard]), target_[hard]
                            )
                            .mean(dim=0)
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        epoch_elikelihoods.append(
                            gaussian_log_likelihood_loss(
                                (mean_[easy], variance_[easy]), target_[easy]
                            )
                            .mean(dim=0)
                            .detach()
                            .cpu()
                            .numpy()
                        )

                optimizer.zero_grad()
                mean_loss.backward()
                self.execute_hooks_for_phase(HookPhase.BACKWARD, model)

                optimizer.step()
                self.execute_hooks_for_phase(HookPhase.OPT_STEP, model)

            self.execute_hooks_for_phase(HookPhase.EPOCH_END, model, dataloader, epoch)

            info_dict["mu_losses"].append(np.mean(epoch_mu_loss))
            info_dict["sigma_losses"].append(np.mean(epoch_sigma_loss))
            if len(epoch_reg_losses) > 0:
                info_dict["reg_losses"].append(np.mean(epoch_reg_losses))
            info_dict["loss"].append(np.mean(epoch_loss))
            info_dict["mses"].append(np.mean(epoch_mses))
            info_dict["stds"].append(np.mean(epoch_stds))
            info_dict["likelihoods"].append(np.mean(epoch_likelihoods))

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
                    eval_mse = self.eval_mse(model, eval_x, eval_y).cpu()
                    eval_likelihood = self.eval_likelihood(model, eval_x, eval_y).cpu()
                    info_dict["val_epochs"].append(epoch)
                    info_dict["val_mses"].append(eval_mse.detach().cpu().numpy())
                    info_dict["val_nlls"].append(eval_likelihood.detach().cpu().numpy())
                    should_early_stop = self.early_stopping(
                        model,
                        logger,
                        epoch,
                        eval_mse=info_dict["val_mses"][-1],
                        eval_likelihood=info_dict["val_nlls"][-1],
                    )

                log_str = (
                    f"Epoch {epoch}: mu loss: {info_dict['mu_losses'][-1]:.4f}, sigma loss: {info_dict['sigma_losses'][-1]:.4f}, "
                    f"rmse: {np.sqrt(info_dict['mses'][-1]):.4f}, std: {info_dict['stds'][-1]:.4f}, NLL: {info_dict['likelihoods'][-1]:.4f}"
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
        mean, std_or_var = model(torch.as_tensor(xs, dtype=torch.float))

        if self.learns_variance:
            variance = std_or_var
        else:
            variance = std_or_var ** 2

        mean, variance = self.denorm(mean, variance)

        return mean.cpu().numpy(), variance.cpu().numpy()

    @torch.no_grad()
    def eval_likelihood(self, model, inp, target):
        inp = inp.to(self.device)
        target = target.to(self.device)
        if model.training:
            model.eval()
        mean, std_or_var = model(inp)

        if self.learns_variance:
            variance = std_or_var
        else:
            variance = std_or_var ** 2

        nll = gaussian_log_likelihood_loss(self.denorm(mean, variance), target)

        return nll.mean()
