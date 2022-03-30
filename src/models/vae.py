import torch
from torch import nn
from torch import distributions as D

from src.models.utils import infer_model_device


class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dims):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims

    def regularizer(self, pred):
        """Compute KLD(q(z | x) || N(0, 1))"""
        z_mean, z_var = pred[-1]

        return -0.5 * torch.sum(1 + z_var.log() - z_mean.pow(2) - z_var, axis=-1)

    def sample_prior(self, n_samples, prior_std=1.0, return_mean=True):
        device = infer_model_device(self)
        z = torch.randn(n_samples, self.latent_dims).to(device) * prior_std

        sample = self.decoder(z)
        sample_mean = sample[0]
        if len(sample) == 2:
            sample_var = sample[1]

        if return_mean:
            return sample_mean
        else:
            return torch.randn_like(sample_mean) * sample_var.sqrt() + sample_mean

    def eval_posterior_predictive(self, x, n_latent_samples=20) -> D.Distribution:
        device = infer_model_device(self)
        x = x.to(device)
        z_mean, z_var = self.encoder(x)

        bs = len(z_mean)
        noise = torch.randn(bs, n_latent_samples, self.latent_dims).to(device)
        z = noise * torch.sqrt(z_var.unsqueeze(1)) + z_mean.unsqueeze(1)

        px_z = self.decoder(z)
        pi = D.Categorical(torch.ones(bs, n_latent_samples, device=device))

        if len(px_z) == 2:
            x_mean, x_var = px_z
            comp = D.Independent(D.Normal(x_mean, x_var.sqrt()), 1)
        elif len(px_z) == 3 or len(px_z) == 4:
            x_mean, x_alpha, x_beta = px_z[:3]
            dof = 2 * x_alpha
            scale = torch.sqrt(x_beta / x_alpha)
            comp = D.Independent(D.StudentT(dof, x_mean, scale), 1)
        else:
            raise ValueError(f"Unknown decoder distribution with {len(px_z)} dims")

        mixture = D.MixtureSameFamily(pi, comp)

        return mixture

    def forward(self, x, with_regularization=False):
        z_mean, z_var = self.encoder(x)

        z = torch.randn_like(z_mean) * torch.sqrt(z_var) + z_mean

        px_z = self.decoder(z)

        if with_regularization:
            if isinstance(px_z, tuple):
                return (*px_z, (z_mean, z_var))
            else:
                return px_z, (z_mean, z_var)
        else:
            return px_z
