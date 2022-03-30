import copy
from abc import abstractmethod

import numpy as np
import torch

from .hooks import HookPhase

IMPLEMENTED_OPTIM = ["adam", "rmsprop", "sgd"]


class BaseTrainer:
    def __init__(
        self,
        optimizer_name="adam",
        lr=1e-4,
        n_epochs=100,
        betas=[0.9, 0.99],
        batch_size=512,
        log_every=5,
        checkpoint=20,
        device="cuda",
        hooks=None,
        momentum=0,
        rms_alpha=0.99,
        target_mean=None,
        target_std=None,
        early_stop_metric=None,
        early_stop_iters=None,
        track_best_metrics=None,
        log_intermediate_results=False,
    ):
        if optimizer_name not in IMPLEMENTED_OPTIM:
            raise ValueError(f"Unkown optmizer name: {optimizer_name}.")

        self.optimizer_name = optimizer_name
        self.model_regularizer = None  # Lazy init
        self.lr = lr
        self.n_epochs = n_epochs
        self.momentum = momentum
        self.batch_size = batch_size
        self.betas = (betas[0], betas[1])
        self.rms_alpha = rms_alpha

        if hooks is not None:
            self.hooks = hooks
        else:
            self.hooks = []

        self.checkpoint = checkpoint
        self.log_every = log_every
        self.log_intermediate_results = log_intermediate_results

        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested, but it's not available")
            self.device = device
        else:
            self.device = "cpu"

        self._target_mean = None
        self._target_std = None

        self.early_stop_metric = early_stop_metric
        self.early_stop_iters = early_stop_iters
        self.early_stop_model = None
        self.early_stop_best = np.inf
        self._early_stop_no_improvements = 0
        self.track_best_metrics = track_best_metrics if track_best_metrics else {}
        self.track_best_models = {}
        self._track_best_no_improvements = {m: 0 for m in self.track_best_metrics}
        self._track_best_values = {m: np.inf for m in self.track_best_metrics}

    def set_target_normalizer(self, mean, std):
        self._target_mean = torch.from_numpy(np.array(mean)).float().to(self.device)
        self._target_std = torch.from_numpy(np.array(std)).float().to(self.device)

    def _get_optimizer(self, params):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(params=params, lr=self.lr)
        elif self.optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                params=params, lr=self.lr, momentum=self.momentum, alpha=self.rms_alpha
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                params=params, lr=self.lr, momentum=self.momentum
            )
        else:
            raise ValueError(f"Unkown optimizer name {self.optimizer_name}.")

        return optimizer

    def val_dataset_to_torch(self, val_dataset):
        val_data = val_dataset[:]
        if len(val_data) == 3:
            eval_x, eval_y, _ = val_data
        else:
            eval_x, eval_y = val_data

        if not isinstance(eval_x, torch.Tensor):
            eval_x = torch.from_numpy(eval_x).float().to(self.device)
        if not isinstance(eval_y, torch.Tensor):
            eval_y = torch.from_numpy(eval_y).float().to(self.device)

        return eval_x, eval_y

    def denorm(self, mean, var=None):
        if self._target_mean is None:
            return mean if var is None else (mean, var)
        elif var is not None:
            mean = mean * self._target_std + self._target_mean
            var = var * (self._target_std ** 2)
            return mean, var
        else:
            return mean * self._target_std + self._target_mean

    @torch.no_grad()
    def eval_mse(self, model, inp, target, pred=None):
        inp = inp.to(self.device)
        target = target.to(self.device)

        if pred is None:
            if model.training:
                model.eval()
            pred = model(inp)

        if isinstance(pred, tuple):
            pred = pred[0]

        pred = self.denorm(pred)
        mse = (pred - target) ** 2

        return mse.mean()

    def early_stopping(self, model, logger, epoch, **kwargs):
        def update(no_improvements, best_value, current_value):
            if current_value < best_value:
                return 0, current_value
            else:
                return no_improvements + 1, best_value

        for metric in self.track_best_metrics:
            if metric not in kwargs:
                logger.info(
                    f"Warning: requested tracking best metric {metric}, but was not found"
                )
                continue
            no_improvements, best_value = update(
                self._track_best_no_improvements[metric],
                self._track_best_values[metric],
                kwargs[metric],
            )
            self._track_best_no_improvements[metric] = no_improvements
            self._track_best_values[metric] = best_value
            if no_improvements == 0:
                print(f"Saving best {metric}: {best_value}")
                self.track_best_models[metric] = copy.deepcopy(model)
                self.save_checkpoint(
                    logger, model, epoch, postfix=f"best_{metric}", save_hooks=False
                )

        if self.early_stop_metric is None:
            return False

        if self.early_stop_metric not in kwargs:
            raise ValueError(
                "Did not find early stopping metric {self.early_stop_metric}"
            )

        current_value = kwargs[self.early_stop_metric]

        self._early_stop_no_improvements, self.early_stop_best = update(
            self._early_stop_no_improvements, self.early_stop_best, current_value
        )
        if self._early_stop_no_improvements == 0:
            self.early_stop_model = copy.deepcopy(model)
            self.save_checkpoint(logger, model, epoch, postfix="best", save_hooks=False)

        should_early_stop = self._early_stop_no_improvements >= self.early_stop_iters

        if should_early_stop:
            early_stop_epoch = epoch - self._early_stop_no_improvements
            self.execute_hooks_for_phase(
                HookPhase.EARLY_STOP,
                self.early_stop_model,
                logger,
                early_stop_epoch,
                self.early_stop_metric,
                self.early_stop_best,
            )

        return should_early_stop

    @abstractmethod
    def train():
        raise NotImplementedError("Method train not implemented.")

    @abstractmethod
    def eval_model(self, model, xs):
        raise NotImplementedError("Method eval_model not implemented.")

    def init_model(self, model):
        model.to(self.device)

        if hasattr(model, "regularizer"):
            self.model_regularizer = model.regularizer

        return model

    def backward(self, step, loss, mean_loss, optimizer):
        mean_loss.backward()

    def _gather_gradients(self, model):
        gradients = []
        for p in model.parameters():
            if p.grad is None:
                gradients.append(torch.zeros_like(p).view(-1))
            else:
                gradients.append(p.grad.view(-1))

        return torch.cat(gradients)

    def _eval_gradient(self, model, optimizer, loss, with_update=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=not with_update)

        gradients = self._gather_gradients(model)

        if with_update:
            with torch.no_grad():
                params_before = [p.cpu().clone() for p in model.parameters()]
                optimizer_state = optimizer.state_dict()

                optimizer.step()

                updates = [
                    (p2 - p1).view(-1)
                    for p1, p2 in zip(params_before, model.parameters())
                ]

                optimizer.load_state_dict(optimizer_state)
                for p_after, p_before in zip(model.parameters(), params_before):
                    p_after.copy_(p_before)

            return gradients, torch.cat(updates)
        else:
            return gradients

    def execute_hooks_for_phase(self, phase: HookPhase, *args, **kwargs):
        for hook in self.hooks:
            if phase in hook.phases:
                hook(phase, *args, **kwargs)

    def log(self, logger, epoch, info_dict=None):
        if info_dict is not None and self.log_intermediate_results:
            logger.log_training_as_dataframe(info_dict)

        self.execute_hooks_for_phase(HookPhase.LOGGING, logger, epoch)

    def save_checkpoint(self, logger, model, epoch, postfix=None, save_hooks=True):
        postfix = str(epoch + 1) if postfix is None else postfix
        logger.save_model(model, f"m_{postfix}")
        if save_hooks and len(self.hooks) > 0:
            logger.save_hooks(self.hooks)
