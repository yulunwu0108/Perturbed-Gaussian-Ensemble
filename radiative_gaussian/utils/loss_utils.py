import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn


def mse(img1, img2, mask=None):
    """MSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_channel = img1.shape[1]
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(1, n_channel)
        mask = torch.where(mask != 0, True, False)

        mse = torch.stack(
            [
                (((img1[i, mask[i]] - img2[i, mask[i]])) ** 2).mean(0, keepdim=True)
                for i in range(img1.shape[0])
            ],
            dim=0,
        )

    else:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse

def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def psnr(img1, img2, mask=None, pixel_max=1.0):
    """PSNR

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
    if mask is not None:
        if torch.isinf(psnr_out).any():
            print(mse_out.mean(), psnr_out.mean())
            psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
            psnr_out = psnr_out[~torch.isinf(psnr_out)]

    return psnr_out