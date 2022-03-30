import math

import torch
import torch.nn as nn
# from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True, **kwargs):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss


_LOG_2PI = math.log(2 * math.pi)


class GaussianLogLikelihoodLoss(nn.Module):
    def __init__(self, beta=0):
        super(GaussianLogLikelihoodLoss, self).__init__()
        self.name = 'NLL'
        self.beta = beta
        if beta > 0:
            self.name += f'_{beta:.2f}'

    def forward(self, input, target, mask=None, interpolate=True, variance=None):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
            if variance is not None:
                variance = nn.functional.interpolate(variance, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
            if variance is not None:
                variance = variance[mask]

        mean = input
        ll = -0.5 * ((target - mean) ** 2 / variance + torch.log(variance) + _LOG_2PI)

        if self.beta > 0:
            weight = variance.detach() ** self.beta
            ll = ll * weight

        # Can not take sum over dimensions, because each batch element has
        # different length due to masking
        return -torch.mean(ll)


class MSELoss(nn.Module):
    def __init__(self, beta=0):
        super(MSELoss, self).__init__()
        self.name = 'MSE'

    def forward(self, input, target, mask=None, interpolate=True, **kwargs):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]

        mse = (target - input) ** 2

        return torch.mean(mse)


class L1Loss(nn.Module):
    def __init__(self, beta=0):
        super(L1Loss, self).__init__()
        self.name = 'L1'

    def forward(self, input, target, mask=None, interpolate=True, **kwargs):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]

        l1 = torch.abs(target - input)

        return torch.mean(l1)
