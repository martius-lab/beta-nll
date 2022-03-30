import math

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from src.models import utils


def init_module(m, w_init, b_init):
    if hasattr(m, "weight") and m.weight is not None and w_init is not None:
        w_init(m.weight)
    if hasattr(m, "bias") and m.bias is not None and b_init is not None:
        b_init(m.bias)


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        outp_dim,
        hidden_dims,
        hidden_activation=nn.ReLU,
        outp_layer=nn.Linear,
        outp_activation=nn.Identity,
        weight_init=None,
        bias_init=None,
        weight_init_last=None,
        bias_init_last=None,
        batchnorm_first=False,
        use_spectral_norm=False,
    ):
        super().__init__()
        self.w_init = weight_init
        self.b_init = bias_init
        self.w_init_last = weight_init_last
        self.b_init_last = bias_init_last

        if batchnorm_first:
            self.input_bn = nn.BatchNorm1d(inp_dim, momentum=0.1, affine=False)
        else:
            self.input_bn = None

        layers = []
        current_dim = inp_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_spectral_norm:
                layers[-1] = nn.utils.spectral_norm(layers[-1])
            layers.append(hidden_activation())
            current_dim = hidden_dim

        layers.append(outp_layer(current_dim, outp_dim))
        if outp_activation is not None:
            layers.append(outp_activation())

        self.layers = nn.Sequential(*layers)
        self.init()

    def init(self):
        self.layers.apply(lambda m: init_module(m, self.w_init, self.b_init))
        self.layers[-2].apply(
            lambda m: init_module(m, self.w_init_last, self.b_init_last)
        )

    def forward(self, inp):
        if self.input_bn is not None:
            inp = self.input_bn(inp)

        return self.layers(inp)


class GaussianLikelihoodHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        outp_dim,
        initial_var=1,
        min_var=1e-8,
        max_var=100,
        mean_scale=1,
        var_scale=1,
        use_spectral_norm_mean=False,
        use_spectral_norm_var=False,
    ):
        super().__init__()
        assert min_var <= initial_var <= max_var

        self.min_var = min_var
        self.max_var = max_var
        self.init_var_offset = np.log(np.exp(initial_var - min_var) - 1)

        self.mean_scale = mean_scale
        self.var_scale = var_scale

        if use_spectral_norm_mean:
            self.mean = nn.utils.spectral_norm(nn.Linear(inp_dim, outp_dim))
        else:
            self.mean = nn.Linear(inp_dim, outp_dim)

        if use_spectral_norm_var:
            self.var = nn.utils.spectral_norm(nn.Linear(inp_dim, outp_dim))
        else:
            self.var = nn.Linear(inp_dim, outp_dim)

    def forward(self, inp):
        mean = self.mean(inp) * self.mean_scale
        var = self.var(inp) * self.var_scale

        var = F.softplus(var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, self.min_var, self.max_var)

        return mean, var


class GaussianLikelihoodFixedVarianceHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        outp_dim,
        fixed_var=1,
        mean_scale=1,
        use_spectral_norm_mean=False,
    ):
        super().__init__()
        self.mean_scale = mean_scale

        if use_spectral_norm_mean:
            self.mean = nn.utils.spectral_norm(nn.Linear(inp_dim, outp_dim))
        else:
            self.mean = nn.Linear(inp_dim, outp_dim)

        self.register_buffer(
            "variance", torch.full((outp_dim,), fixed_var, dtype=torch.float32)
        )

    def forward(self, inp):
        mean = self.mean(inp) * self.mean_scale
        var = torch.broadcast_tensors(self.variance, mean)[0]
        return mean, var


class LinearHead(nn.Module):
    def __init__(
        self, inp_dim, outp_dim,
    ):
        super().__init__()

        self.l = nn.Linear(inp_dim, outp_dim)

    def forward(self, inp):
        return self.l(inp)


_MIN_VARIANCE = 1e-8
_INIT_VARIANCE = 1
_MAX_VARIANCE = 1000
_MIN_ALPHA = 1.001
_INIT_ALPHA = math.log(1 + math.exp(0)) + _MIN_ALPHA
_MAX_ALPHA = 1000
_MIN_BETA = (_MIN_ALPHA - 1) * _MIN_VARIANCE
_INIT_BETA = _INIT_ALPHA - 1 * _INIT_VARIANCE
_MAX_BETA = (_MAX_ALPHA - 1) * _MAX_VARIANCE


class NormalGammaHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        outp_dim,
        initial_alpha=_INIT_ALPHA,
        min_alpha=_MIN_ALPHA,
        max_alpha=_MAX_ALPHA,
        initial_beta=_INIT_BETA,
        min_beta=_MIN_BETA,
        max_beta=_MAX_BETA,
        mean_scale=1,
        alpha_scale=1,
        beta_scale=1,
    ):
        super().__init__()
        assert min_alpha <= initial_alpha <= max_alpha
        assert min_beta <= initial_beta <= max_beta

        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.init_alpha_offset = utils.softmax_inverse(initial_alpha - min_alpha)

        self.min_beta = min_beta
        self.max_beta = max_beta
        self.init_beta_offset = utils.softmax_inverse(initial_beta - min_beta)

        self.mean_scale = mean_scale
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale

        self.layer = nn.Linear(inp_dim, 3 * outp_dim)

    def forward(self, inp):
        x = self.layer(inp)
        N = x.shape[-1] // 3

        mean = x[..., :N] * self.mean_scale
        alpha = x[..., N : 2 * N] * self.alpha_scale
        beta = x[..., 2 * N :] * self.beta_scale

        alpha = F.softplus(alpha + self.init_alpha_offset) + self.min_alpha
        alpha = torch.clamp(alpha, self.min_alpha, self.max_alpha)

        beta = F.softplus(beta + self.init_beta_offset) + self.min_beta
        beta = torch.clamp(beta, self.min_beta, self.max_beta)

        return mean, alpha, beta
