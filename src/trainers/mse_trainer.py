from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F

from .base_trainer import BaseTrainer
from .hooks import HookPhase


class MSETrainer(BaseTrainer):
    def __init__(
        self,
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
        bce_loss=False,
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
        if bce_loss:
            self.loss_fn = lambda pred, target: (
                F.binary_cross_entropy(pred, target, reduction="none").sum(dim=-1)
            )
        else:
            self.loss_fn = lambda pred, target: ((target - pred) ** 2).mean(dim=-1)
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
        optimizer = self._get_optimizer(model.parameters())
        should_early_stop = False
        self.execute_hooks_for_phase(HookPhase.INIT, model, optimizer, dataloader)

        for epoch in range(self.n_epochs):
            self.execute_hooks_for_phase(
                HookPhase.EPOCH_START, model, dataloader, epoch
            )
            model.train()

            epoch_losses = []
            epoch_mses, epoch_stds = [], []
            for idx, batch in enumerate(dataloader):
                global_step = epoch * len(dataloader) + idx

                inp = batch[0].float().unsqueeze(1).to(self.device)
                target = batch[1].float().unsqueeze(1).to(self.device)

                self.execute_hooks_for_phase(
                    HookPhase.MODEL_FORWARD, model, inp, target
                )
                pred = model(inp)
                loss = self.loss_fn(pred, target)
                self.execute_hooks_for_phase(
                    HookPhase.MODEL_FORWARD_END, model, inp, pred, loss, target
                )

                mean_loss = loss.mean()

                with torch.no_grad():
                    pred_ = self.denorm(pred)
                    target_ = self.denorm(target)
                    mse = (target_ - pred_) ** 2
                    epoch_losses.append(loss.detach().cpu().numpy())
                    epoch_mses.append(mse.mean(dim=-1).detach().cpu().numpy())
                    epoch_stds.append(mse.std().detach().cpu().numpy())

                optimizer.zero_grad()
                self.backward(global_step, loss, mean_loss, optimizer)
                self.execute_hooks_for_phase(HookPhase.BACKWARD, model)

                optimizer.step()
                self.execute_hooks_for_phase(HookPhase.OPT_STEP, model)

            self.execute_hooks_for_phase(HookPhase.EPOCH_END, model, dataloader, epoch)

            info_dict["losses"].append(np.mean(epoch_losses))
            info_dict["mses"].append(np.mean(epoch_mses))
            info_dict["stds"].append(np.mean(epoch_stds))

            if self.log_every > 0 and (
                epoch % self.log_every == 0 or epoch == self.n_epochs - 1
            ):
                if eval_x is not None:
                    eval_mse = self.eval_mse(model, eval_x, eval_y).cpu()
                    info_dict["val_epochs"].append(epoch)
                    info_dict["val_mses"].append(eval_mse.detach().cpu().numpy())
                    should_early_stop = self.early_stopping(
                        model, logger, epoch, eval_mse=info_dict["val_mses"][-1]
                    )

                log_str = (
                    f"Epoch {epoch}: {info_dict['losses'][-1]:.4f}, rmse: {np.sqrt(info_dict['mses'][-1]):.4f}, "
                    f"std: {info_dict['stds'][-1]:.4f}"
                )
                if eval_x is not None:
                    log_str += f", eval rmse: {np.sqrt(eval_mse):.4f}"
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
        with torch.no_grad():
            pred = model(torch.as_tensor(xs, dtype=torch.float))

        pred = self.denorm(pred)

        return pred.cpu().numpy(), np.zeros_like(pred.cpu().numpy())
