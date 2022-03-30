import math

import numpy as np
import torch
from torch.nn import init


_LOG_2PI = math.log(2 * math.pi)
_LOG_PI = math.log(math.pi)


def eval_accuracy(ys, std_eps, y_mean, y_var):
    mse = np.mean((y_mean - ys) ** 2)
    var_err = np.mean(np.abs(np.sqrt(y_var) - std_eps))
    return mse, var_err


def softmax_inverse(x):
    return np.log(np.exp(x) - 1)


def gaussian_log_likelihood_loss(pred, target):
    mean, var = pred
    ll = -0.5 * ((target - mean) ** 2 / var + torch.log(var) + _LOG_2PI)

    return -torch.sum(ll, axis=-1)


def gaussian_beta_log_likelihood_loss(pred, target, beta=1):
    mean, var = pred
    ll = -0.5 * ((target - mean) ** 2 / var + torch.log(var) + _LOG_2PI)
    weight = pred[1].detach() ** beta

    return -torch.sum(ll * weight, axis=-1)


def student_t_from_inv_gamma_nll(pred, target):
    # Parametrized in terms of inverse gamma dist on variance
    loc, alpha, beta = pred
    dof = 2 * alpha
    scale = torch.sqrt(beta / alpha)

    return student_t_nll(dof, loc, scale, target)


def student_t_nll(dof, loc, scale, target):
    # Adapted from torch.distributions.studentT
    y = (target - loc) / scale
    Z = (
        scale.log()
        + 0.5 * dof.log()
        + 0.5 * _LOG_PI
        + torch.lgamma(0.5 * dof)
        - torch.lgamma(0.5 * (dof + 1.0))
    )
    log_prob = -0.5 * (dof + 1.0) * torch.log1p(y ** 2.0 / dof) - Z

    return -torch.sum(log_prob, axis=-1)


def get_activation(opts):
    if opts.hidden_activation == "tanh":
        activation = torch.nn.Tanh
    elif opts.hidden_activation == "relu":
        activation = torch.nn.ReLU
    elif opts.hidden_activation == "lrelu_0.01":
        activation = lambda: torch.nn.LeakyReLU(0.01)
    elif opts.hidden_activation == "lrelu_0.1":
        activation = lambda: torch.nn.LeakyReLU(0.1)
    elif opts.hidden_activation == "elu":
        activation = torch.nn.ELU
    else:
        raise ValueError(f"Unknown activation function name {opts.hidden_activation}.")

    return activation


def get_weight_init(opts):
    if opts.weight_init == "xavier_uniform":
        w_init = init.xavier_uniform_
    elif opts.weight_init == "orthogonal":
        w_init = init.orthogonal_
    elif opts.weight_init == "lecun":
        w_init = lambda w: init.kaiming_uniform_(w, nonlinearity="linear")
    else:
        raise ValueError(f"Unknown weight init {opts.weight_init}.")

    return w_init


def get_layers(model):
    from .networks import GaussianLikelihoodHead

    layers = []
    for l in model.layers:
        if isinstance(l, torch.nn.Linear):
            layers.append(l)
        if isinstance(l, GaussianLikelihoodHead):
            layers += [l.mean, l.var]

    return layers


def extract_flatten_weights(layers):
    weights_vector = []
    for l in layers:
        weights_vector.append(l.weight.detach().flatten())

    return weights_vector


def infer_model_device(model):
    param = next(iter(model.parameters()))
    return param.device
