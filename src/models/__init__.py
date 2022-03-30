import torch

from src.models import utils
from src.models.networks import (
    MLP,
    GaussianLikelihoodHead,
)
from src.models.vae import VAE


def get_model(inp_dim, outp_dim, head, opts):

    if opts.model_type == "MLP":
        model = MLP(
            inp_dim=inp_dim,
            outp_dim=outp_dim,
            hidden_dims=opts.hidden_dims,
            hidden_activation=utils.get_activation(opts),
            weight_init=utils.get_weight_init(opts),
            bias_init=torch.nn.init.zeros_,
            outp_layer=head,
            batchnorm_first=opts.batchnorm_first,
            use_spectral_norm=opts.spectral_norm,
        )
    elif opts.model_type == "VAE":
        latent_head = lambda inp, outp: GaussianLikelihoodHead(
            inp, outp, initial_var=1, max_var=100
        )
        encoder = MLP(
            inp_dim=inp_dim,
            outp_dim=opts.latent_dims,
            hidden_dims=opts.hidden_dims,
            hidden_activation=utils.get_activation(opts),
            weight_init=utils.get_weight_init(opts),
            bias_init=torch.nn.init.zeros_,
            outp_layer=latent_head,
            batchnorm_first=opts.batchnorm_first,
            use_spectral_norm=opts.spectral_norm,
        )
        decoder = MLP(
            inp_dim=opts.latent_dims,
            outp_dim=outp_dim,
            hidden_dims=list(reversed(opts.hidden_dims)),
            hidden_activation=utils.get_activation(opts),
            weight_init=utils.get_weight_init(opts),
            bias_init=torch.nn.init.zeros_,
            outp_layer=head,
            batchnorm_first=opts.batchnorm_first,
            use_spectral_norm=opts.spectral_norm,
        )
        model = VAE(encoder, decoder, opts.latent_dims)
    else:
        raise ValueError(f"Unknown model type {opts.model_type}")

    return model
