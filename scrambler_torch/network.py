"""Scrambler ResNet: one-hot sequence -> positive per-position importance scores.

Mirrors `load_scrambler_network` / `make_resblock` of the original Keras code
(1D signals only). Tensors are (batch, length, channels) at the module boundary
and (batch, channels, length) internally.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal.windows import gaussian

KERAS_EPS = 1e-7  # keras.backend.epsilon()


def glorot_normal_(w):
    """Keras 2.2 'glorot_normal': truncated normal, std = sqrt(2 / (fan_in + fan_out)) / .8796."""
    receptive = w[0][0].numel() if w.dim() > 2 else 1
    fan_in, fan_out = w.shape[1] * receptive, w.shape[0] * receptive
    std = math.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
    with torch.no_grad():
        return nn.init.trunc_normal_(w, 0.0, std, -2 * std, 2 * std)


def conv1d(c_in, c_out, kernel_size=1, dilation=1, bias=True):
    conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation, padding="same", bias=bias)
    glorot_normal_(conv.weight)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def make_norm(norm_mode, channels):
    if norm_mode == "instance":
        return nn.InstanceNorm1d(channels, eps=1e-6, affine=True)
    if norm_mode == "batch":  # Keras BatchNormalization defaults
        return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
    return nn.Identity()


class ResBlock(nn.Module):
    """norm -> relu -> conv -> norm -> relu -> conv (-> dropout) + skip."""

    def __init__(self, channels, kernel_size, dilation=1, drop_rate=0.0, norm_mode="instance"):
        super().__init__()
        self.norm0 = make_norm(norm_mode, channels)
        self.conv0 = conv1d(channels, channels, kernel_size, dilation)
        self.norm1 = make_norm(norm_mode, channels)
        self.conv1 = conv1d(channels, channels, kernel_size, dilation)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        h = self.conv0(F.relu(self.norm0(x)))
        h = self.drop(self.conv1(F.relu(self.norm1(h))))
        return h + x


def mask_dropout(mask, drop_scales=(1, 5), min_rate=0.0, max_rate=0.5):
    """Random multi-scale block dropout of a (B, 1, L) mask (training only).

    One drop rate per example is split across scales by a random softmax; at
    scale s, blocks of s consecutive positions are dropped together.
    """
    b, _, length = mask.shape
    rates = torch.empty(b, 1, 1, device=mask.device).uniform_(min_rate, max_rate)
    if len(drop_scales) > 1:
        scale_probs = torch.softmax(torch.empty(b, len(drop_scales), device=mask.device).uniform_(-5.0, 5.0), dim=1)
    for i, s in enumerate(drop_scales):
        r = rates * scale_probs[:, i, None, None] if len(drop_scales) > 1 else rates
        keep = torch.rand(b, 1, -(-length // s), device=mask.device) >= r
        mask = mask * keep.repeat_interleave(s, dim=2)[..., :length].float()
    return mask


class ScramblerNet(nn.Module):
    """Dilated ResNet predicting importance scores S(X) > 0 for each position.

    Argument names follow the original `network_config` keys: `n_out_channels`
    is the alphabet size, `n_channels` the hidden width.
    forward(x, drop=None) with x: (B, L, C) one-hot, drop: optional (B, L) mask
    (only used with mask_dropout=True). Returns scores (B, L).
    """

    def __init__(self, n_out_channels=4, n_groups=1, n_resblocks_per_group=4, n_channels=32, window_size=8,
                 dilation_rates=(1,), drop_rate=0.0, norm_mode="instance", mask_smoothing=False,
                 mask_smoothing_window_size=7, mask_smoothing_std=1.5, mask_dropout=False,
                 mask_drop_scales=(1, 5), mask_min_drop_rate=0.0, mask_max_drop_rate=0.5):
        super().__init__()
        self.mask_dropout = mask_dropout
        self.mask_drop_kwargs = dict(drop_scales=tuple(mask_drop_scales), min_rate=mask_min_drop_rate,
                                     max_rate=mask_max_drop_rate)
        self.conv0 = conv1d(n_out_channels + int(mask_dropout), n_channels)
        self.skip_convs = nn.ModuleList(conv1d(n_channels, n_channels) for _ in range(n_groups))
        self.groups = nn.ModuleList(
            nn.Sequential(*[ResBlock(n_channels, window_size, dilation_rates[g], drop_rate, norm_mode)
                            for _ in range(n_resblocks_per_group)])
            for g in range(n_groups))
        self.last_block_conv = conv1d(n_channels, n_channels)
        self.final_conv = conv1d(n_channels, 1)
        if mask_smoothing:
            self.register_buffer("smooth_kernel", gaussian_kernel(mask_smoothing_window_size, mask_smoothing_std))
        else:
            self.smooth_kernel = None

    def forward(self, x, drop=None):
        h = x.transpose(1, 2)
        if self.mask_dropout:
            drop = torch.ones_like(h[:, :1]) if drop is None else drop[:, None, :].to(h)
            if self.training:
                drop = mask_dropout(drop, **self.mask_drop_kwargs)
            h = torch.cat([h, drop], dim=1)
        h = self.conv0(h)
        skips = []
        for skip_conv, group in zip(self.skip_convs, self.groups):
            skips.append(skip_conv(h))
            h = group(h)
        h = self.last_block_conv(h)
        for s in skips:
            h = h + s
        scores = F.softplus(self.final_conv(h))
        if self.mask_dropout:
            scores = scores * drop
        if self.smooth_kernel is not None:
            scores = smooth(scores, self.smooth_kernel)
        return scores[:, 0]


def smooth(scores, kernel):
    """'same'-padded fixed-kernel smoothing of (B, 1, L) scores."""
    return F.conv1d(scores, kernel, padding="same")


def gaussian_kernel(window_size, std):
    return torch.tensor(gaussian(window_size, std), dtype=torch.float32).view(1, 1, -1)


def scale_logits(x, scores, scrambler_mode="inclusion"):
    """Map one-hot x to {-1, 1} logits and scale by scores (inclusion) or 1/scores (occlusion)."""
    logits = 2.0 * x - 1.0
    if scrambler_mode == "inclusion":
        return logits * scores[..., None]
    if scrambler_mode == "occlusion":
        return logits / scores.clamp_min(KERAS_EPS)[..., None]
    raise ValueError(f"unknown scrambler_mode {scrambler_mode!r}")
