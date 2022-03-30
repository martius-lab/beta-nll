"""Implements algorithms from

Stirn & Knowles, 2020: "Variational Variance: Simple, Reliable, Calibrated
Heteroscedastic Noise Variance Parameterization".
"""
import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import gamma
from torch.nn import functional as F

from src.models import utils
from src.models.networks import NormalGammaHead

_LOG_2PI = math.log(2 * math.pi)


def exp_gaussian_nll_under_gamma_prec(mean, alpha, beta, target):
    """Compute expected likelihood of log normal distribution
    under a Gamma distributed precision

    See below Eq. 1 in paper
    """
    return -0.5 * (
        torch.digamma(alpha)
        - beta.log()
        - _LOG_2PI
        - alpha * ((target - mean) ** 2) / beta
    ).sum(axis=-1)


class NormalGammaPiHead(nn.Module):
    def __init__(self, inp_dim, outp_dim, n_components, **kwargs):
        super().__init__()
        self.normal_gamma_head = NormalGammaHead(inp_dim, outp_dim, **kwargs)
        self.pi = nn.Linear(inp_dim, outp_dim * n_components)

    def forward(self, inp):
        mean, alpha, beta = self.normal_gamma_head(inp)
        pi_logits = self.pi(inp).view(mean.shape[0], mean.shape[-1], -1)

        return mean, alpha, beta, pi_logits


def kl_gamma_mixture_of_gammas(dist, mixture_logits, mixture_dists, n_mc_samples):
    # dist: B x D
    # mixture_dists: D x C
    n_components = mixture_dists.rate.shape[-1]

    prec_samples = dist.rsample((n_mc_samples,))  # S x B x D
    prec_samples = prec_samples.unsqueeze(-1)
    prec_samples = prec_samples.expand((-1, -1, -1, n_components))

    ll_prior_c = mixture_dists.log_prob(prec_samples)  # S x B x D x C
    log_pi = F.log_softmax(mixture_logits, dim=-1)  # B x D x C
    ll_prior = torch.logsumexp(log_pi + ll_prior_c, axis=-1)
    ll_prior = torch.mean(ll_prior, axis=0)  # Mean over MC samples

    kld = -dist.entropy() - ll_prior  # B x D

    return kld.sum(axis=-1)


def get_vbem_standard_init(n_dims):
    params = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    alpha_beta = np.array(tuple(itertools.product(params, params)), dtype=np.float32).T
    alphas = utils.softmax_inverse(alpha_beta[0])
    alphas = torch.from_numpy(alphas).unsqueeze(0).repeat(n_dims, 1)
    betas = utils.softmax_inverse(alpha_beta[1])
    betas = torch.from_numpy(betas).unsqueeze(0).repeat(n_dims, 1)
    return alphas, betas


def get_vbem_star_standard_init(n_dims, n_components=100):
    # Initialize prior params on Unif[-3, 3]
    alphas = nn.Parameter(torch.rand(n_dims, n_components) * 6 - 3)
    betas = nn.Parameter(torch.rand(n_dims, n_components) * 6 - 3)
    return alphas, betas


class VBEMRegularizer(nn.Module):
    def __init__(
        self, prior_alphas, prior_betas, n_mc_samples=20, prior_trainable=True
    ):
        super().__init__()
        self.n_mc_samples = n_mc_samples

        if prior_trainable:
            self.prior_alphas = nn.Parameter(prior_alphas)
            self.prior_betas = nn.Parameter(prior_betas)
        else:
            self.register_buffer("prior_alphas", prior_alphas)
            self.register_buffer("prior_betas", prior_betas)

    def forward(self, pred):
        assert len(pred) == 4
        _, alpha, beta, pi_logits = pred

        dist = gamma.Gamma(alpha, beta)
        prior_dist = gamma.Gamma(
            F.softplus(self.prior_alphas), F.softplus(self.prior_betas)
        )

        return kl_gamma_mixture_of_gammas(
            dist, pi_logits, prior_dist, self.n_mc_samples
        )


class xVAMPRegularizer(nn.Module):
    def __init__(self, model, prior_inputs, n_mc_samples=20, prior_trainable=True):
        super().__init__()
        # Store in lambda to avoid having duplicate parameters in optimizer
        self.model = lambda *args, **kwargs: model(*args, **kwargs)
        self.n_mc_samples = n_mc_samples

        if prior_trainable:
            self.prior_inputs = nn.Parameter(prior_inputs.clone())
        else:
            self.register_buffer("prior_inputs", prior_inputs.clone())

    def forward(self, pred):
        assert len(pred) == 4
        _, alpha, beta, pi_logits = pred
        dist = gamma.Gamma(alpha, beta)

        _, prior_alphas, prior_betas, _ = self.model(self.prior_inputs)
        prior_dist = gamma.Gamma(prior_alphas.T, prior_betas.T)  # D x C

        return kl_gamma_mixture_of_gammas(
            dist, pi_logits, prior_dist, self.n_mc_samples
        )
