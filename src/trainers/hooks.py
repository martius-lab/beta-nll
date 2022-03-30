import enum
from typing import Set

import numpy as np
import torch
import torchvision

from src.models.utils import infer_model_device


class HookPhase(enum.Enum):
    INIT = 0
    END = 1
    EPOCH_START = 2
    EPOCH_END = 3
    LOGGING = 4
    MODEL_FORWARD = 5
    MODEL_FORWARD_END = 6
    BACKWARD = 7
    OPT_STEP = 8
    EARLY_STOP = 9
    TEST = 10


class BaseHook:
    def __init__(self, phases):
        assert isinstance(phases, set)
        self._phases = phases

    @property
    def phases(self) -> Set[HookPhase]:
        """Specifies hook phases that this hook should be called on

        Example:

        def phases(self):
            return {HookPhase.EPOCH_START, HookPhase.EPOCH_END}
        """
        raise NotImplementedError("Overwrite method in subclass.")

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, phase, *args, **kwargs):
        if phase == HookPhase.INIT:
            return self.on_init(*args, **kwargs)
        elif phase == HookPhase.END:
            return self.on_end(*args, **kwargs)
        elif phase == HookPhase.EPOCH_START:
            return self.on_epoch_start(*args, **kwargs)
        elif phase == HookPhase.EPOCH_END:
            return self.on_epoch_end(*args, **kwargs)
        elif phase == HookPhase.LOGGING:
            return self.on_logging(*args, **kwargs)
        elif phase == HookPhase.MODEL_FORWARD:
            return self.on_model_forward(*args, **kwargs)
        elif phase == HookPhase.MODEL_FORWARD_END:
            return self.on_model_forward_end(*args, **kwargs)
        elif phase == HookPhase.BACKWARD:
            return self.on_backward(*args, **kwargs)
        elif phase == HookPhase.OPT_STEP:
            return self.on_opt_step(*args, **kwargs)
        elif phase == HookPhase.EARLY_STOP:
            return self.on_early_stop(*args, **kwargs)
        elif phase == HookPhase.TEST:
            return self.on_test(*args, **kwargs)
        else:
            raise ValueError(f"Unknown phase {phase}")

    def on_init(self, model, optimizer, dataloader):
        """Before starting training"""
        raise NotImplementedError("Overwrite method in subclass.")

    def on_end(self, model, info_dict):
        """After finishing training"""
        raise NotImplementedError("Overwrite method in subclass.")

    def on_epoch_start(self, model, dataloader, epoch):
        """Before starting an epoch"""
        raise NotImplementedError("Overwrite method in subclass.")

    def on_epoch_end(self, model, dataloader, epoch):
        """After finishing an epoch"""
        raise NotImplementedError("Overwrite method in subclass.")

    def on_logging(self, logger, epoch):
        """When logging output occurs"""
        raise NotImplementedError("Overwrite method in subclass.")

    def on_model_forward(self, model, inp, target):
        """Before executing model"""
        raise NotImplementedError("Overwrite method in subclass.")

    def on_model_forward_end(self, model, inp, outp, loss, target):
        """After executing model"""
        raise NotImplementedError("Overwrite method in subclass.")

    def on_backward(self, model):
        """After executing backward"""
        raise NotImplementedError("Overwrite method in subclass.")

    def on_opt_step(self, model):
        """After executing optimizer step"""
        raise NotImplementedError("Overwrite method in subclass.")

    def on_early_stop(self, model, logger, epoch, metric, value):
        """After early stopping is triggered

        model: Model which triggered early stopping
        epoch: Epoch from which early stopping model is from
        metric: Early stopping metric
        value: Best value of the metric
        """
        raise NotImplementedError("Overwrite method in subclass.")

    def on_test(
        self, model, model_name, test_inputs, test_targets, metric_dict, logger
    ):
        raise NotImplementedError("Overwrite method in subclass.")

    def state_dict(self):
        return {}


class EpochHook(BaseHook):
    def __init__(self, eval_every, last_epoch, stop_after=None):
        super().__init__({HookPhase.INIT, HookPhase.EPOCH_END})
        self.eval_every = eval_every
        self.last_epoch = last_epoch
        self.stop_after = stop_after if stop_after else [last_epoch]
        self.data_per_epoch = {}
        assert self.stop_after == sorted(self.stop_after)
        assert len(self.eval_every) == len(self.stop_after)

    def forward(self, model):
        raise NotImplementedError("Overwrite method in subclass.")

    def on_init(self, model, optimizer, dataloader):
        self.data_per_epoch["-1"] = self.forward(model)

    def on_epoch_end(self, model, dataloader, epoch):
        if epoch == self.last_epoch - 1:
            self.data_per_epoch[str(epoch)] = self.forward(model)
            return
        elif epoch > self.stop_after[-1]:
            return

        eval_every = [
            self.eval_every[idx]
            for idx, stop in enumerate(self.stop_after)
            if epoch <= stop
        ][0]
        if epoch % eval_every == 0:
            self.data_per_epoch[str(epoch)] = self.forward(model)

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update({"data_per_epoch": self.data_per_epoch})
        return state_dict


class EvalHook(EpochHook):
    def __init__(
        self, eval_points, eval_every, last_epoch, stop_after=None, targets=None
    ):
        super().__init__(eval_every, last_epoch, stop_after)
        if isinstance(eval_points, np.ndarray):
            eval_points = torch.from_numpy(eval_points).to(torch.float)
        self.eval_points = eval_points
        if targets is not None and isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).to(torch.float)
        self.targets = targets

    def forward(self, model):
        model.eval()
        with torch.no_grad():
            res = model(self.eval_points)
        if isinstance(res, tuple):
            mean, variance = res
        else:
            mean = res
            variance = torch.zeros_like(mean)

        return self.eval_points.numpy(), mean.numpy(), variance.numpy()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update({"eval_points": self.eval_points})
        state_dict.update({"targets": self.targets})
        return state_dict


def batch_jacobian(f, x):
    def f_sum(x):
        return torch.sum(f(x), axis=0)

    return torch.autograd.functional.jacobian(f_sum, x).permute(1, 0, 2)


class JacobianVarianceHook(EpochHook):
    def __init__(
        self,
        eval_points,
        granularities,
        target_layer,
        eval_every,
        last_epoch,
        stop_after=None,
    ):
        super().__init__(eval_every, last_epoch, stop_after)
        if isinstance(eval_points, np.ndarray):
            eval_points = torch.from_numpy(eval_points).to(torch.float)
        assert eval_points.ndim == 2
        self.eval_points = eval_points
        self.granularities = granularities
        self.target_layer = target_layer
        with torch.no_grad():
            self.pairwise_distances = (
                torch.cdist(self.eval_points[None], self.eval_points[None], p=2)
                .squeeze(0)
                .numpy()
            )
            self.masks_per_granularity = {
                g: self.pairwise_distances <= g for g in self.granularities
            }

    def forward(self, model):
        model.eval()

        def get_features(X):
            for idx, layer in enumerate(model.layers):
                X = layer(X)
                if idx == self.target_layer:
                    break

            return X

        jacobian = batch_jacobian(get_features, self.eval_points)
        jacobian = jacobian.reshape(len(jacobian), -1).numpy()

        variance_per_granularity = {}
        for granularity in self.granularities:
            variances = []
            for mask in self.masks_per_granularity[granularity]:
                variances.append(np.mean(np.var(jacobian[mask], axis=0)))
            variance_per_granularity[granularity] = np.array(variances)

        return variance_per_granularity

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update(
            {
                "eval_points": self.eval_points,
                "granularities": self.granularities,
                "target_layer": self.target_layer,
            }
        )
        return state_dict


class VAEVisualizationHook(BaseHook):
    def __init__(self, eval_every, last_epoch, image_size, n_samples=10):
        super().__init__(
            {
                HookPhase.INIT,
                HookPhase.EPOCH_END,
                HookPhase.LOGGING,
                HookPhase.EARLY_STOP,
                HookPhase.TEST,
            }
        )
        self.eval_every = eval_every[0]
        self.last_epoch = last_epoch
        self.n_samples = n_samples
        self.examples = None
        self.size = image_size
        self._grids_to_save = {}
        self._device = None

    @torch.no_grad()
    def forward(self, model, inputs):
        n_samples = len(inputs)
        dist = model.eval_posterior_predictive(inputs)
        mean = dist.mean.clamp(0, 1)
        std = dist.stddev.clamp(0, 1)

        posterior_sample = dist.sample().clamp(0, 1)
        prior_sample = model.sample_prior(n_samples).clamp(0, 1)

        grid = torch.cat(
            (
                inputs.view(-1, *self.size),
                mean.view(-1, *self.size),
                std.view(-1, *self.size),
                posterior_sample.view(-1, *self.size),
                prior_sample.view(-1, *self.size),
            ),
            axis=0,
        )
        grid = torchvision.utils.make_grid(grid, nrow=n_samples)

        return grid.cpu().numpy()

    def on_init(self, model, optimizer, dataloader):
        self._device = infer_model_device(model)
        batch = next(iter(dataloader))
        self.examples = batch[0][: self.n_samples].to(self._device)

        self._grids_to_save[-1] = self.forward(model, self.examples)

    def on_epoch_end(self, model, dataloader, epoch):
        if epoch == 0 or epoch % self.eval_every == 0 or epoch - 1 == self.last_epoch:
            self._grids_to_save[epoch] = self.forward(model, self.examples)

    def on_logging(self, logger, epoch):
        """When logging output occurs"""
        for epoch, grid in self._grids_to_save.items():
            logger.log_image(f"vis_{epoch}", grid)

        self._grids_to_save = {}

    def on_early_stop(self, model, logger, epoch, metric, value):
        grid = self.forward(model, self.examples)
        logger.log_image(f"vis_early_stop_{metric}_{epoch}", grid)

    def on_test(
        self, model, model_name, test_inputs, test_targets, metric_dict, logger
    ):
        rng = np.random.RandomState(43)
        idxs = rng.permutation(len(test_inputs))[: self.n_samples]
        grid = self.forward(model, test_inputs[idxs].to(self._device))
        logger.log_image(f"vis_test_{model_name}", grid)


class TestPosteriorPredictiveMetricsHook(BaseHook):
    def __init__(self):
        super().__init__({HookPhase.TEST})

    def on_test(
        self, model, model_name, test_inputs, test_targets, metric_dict, logger
    ):
        batch_size = 500
        device = infer_model_device(model)

        log_probs = []
        mses = []
        for idx in range(0, len(test_inputs), batch_size):
            inputs = test_inputs[idx : idx + batch_size].to(device)
            targets = test_targets[idx : idx + batch_size].to(device)

            with torch.no_grad():
                dist = model.eval_posterior_predictive(inputs)
                log_prob = dist.log_prob(targets).cpu().numpy()
                mse = ((targets - dist.mean) ** 2).mean(dim=-1).cpu().numpy()

            log_probs.append(log_prob)
            mses.append(mse)

        log_prob = np.mean(np.concatenate(log_probs))
        mse = np.mean(np.concatenate(mses))
        metric_dict[f"test_{model_name}_posterior_nll"] = [-log_prob]
        metric_dict[f"test_{model_name}_posterior_mse"] = [mse]

        logger.info(
            (
                f"Model {model_name}: Test Posterior RMSE: "
                f"{np.sqrt(mse):.4f}, Test Posterior NLL: {-log_prob:.4f}"
            )
        )
