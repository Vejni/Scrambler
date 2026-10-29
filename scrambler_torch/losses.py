"""Scrambler objectives (per-example losses, shape (B,)).

Prediction losses take y_true (B, S, K) (reference prediction broadcast over the
S samples) and y_pred (B, S, K) (predictions on scrambled samples). Clipping
constants follow the Keras originals (epsilon = 1e-7).
"""
import math

import torch

EPS = 1e-7


def mse(y_true, y_pred):
    return ((y_true[..., 0] - y_pred[..., 0]) ** 2).mean(-1)


def linear_max_nll(y_true, y_pred):
    return -y_pred[..., 0].mean(-1)


def linear_min_nll(y_true, y_pred):
    return y_pred[..., 0].mean(-1)


def softmax_kl_divergence(y_true, y_pred):
    y_true, y_pred = y_true.clamp(EPS, 1 - EPS), y_pred.clamp(EPS, 1 - EPS)
    return (y_true * torch.log(y_true / y_pred)).sum(-1).mean(-1)


def _bernoulli_kl(p, q):
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


def sigmoid_kl_divergence(y_true, y_pred):
    p, q = y_true[..., 0].clamp(EPS, 1 - EPS), y_pred[..., 0].clamp(EPS, 1 - EPS)
    return _bernoulli_kl(p, q).mean(-1)


def symmetric_sigmoid_kl_divergence(y_true, y_pred):
    p, q = y_true[..., 0].clamp(EPS, 1 - EPS), y_pred[..., 0].clamp(EPS, 1 - EPS)
    return _bernoulli_kl(p, q).mean(-1) + _bernoulli_kl(q, p).mean(-1)


def sigmoid_max_nll(y_true, y_pred):
    return (-torch.log(y_pred[..., 0].clamp(EPS, 1.0))).mean(-1)


def sigmoid_min_nll(y_true, y_pred):
    return (-torch.log(1.0 - y_pred[..., 0].clamp(0.0, 1 - EPS))).mean(-1)


def get_nll_loss(nll_mode="reconstruction", predictor_task="classification", n_classes=1):
    """Same dispatch as the original `Scrambler.train`."""
    table = {
        ("reconstruction", "classification"): softmax_kl_divergence if n_classes > 1 else sigmoid_kl_divergence,
        ("reconstruction", "classification_sym"): symmetric_sigmoid_kl_divergence,
        ("maximization", "classification"): sigmoid_max_nll,
        ("minimization", "classification"): sigmoid_min_nll,
        ("reconstruction", "regression"): mse,
        ("maximization", "regression"): linear_max_nll,
        ("minimization", "regression"): linear_min_nll,
    }
    try:
        return table[(nll_mode, predictor_task)]
    except KeyError:
        raise ValueError(f"unsupported nll_mode/predictor_task: {nll_mode!r}/{predictor_task!r}") from None


def mean_conservation(pwm, mask, background):
    """Mean per-position KL(PSSM || background) in bits over free positions. Inputs (B, L, C)."""
    conservation = (pwm * torch.log(pwm.clamp(EPS, 1 - EPS) / background)).sum(-1) / math.log(2.0)
    free = mask.amax(-1)
    return (conservation * free).sum(-1) / free.sum(-1)


def target_entropy_loss(pwm, mask, background, target_bits=0.0):
    return (mean_conservation(pwm, mask, background) - target_bits) ** 2


def margin_entropy_loss(pwm, mask, background, max_bits=0.0):
    return torch.relu(mean_conservation(pwm, mask, background) - max_bits)


def get_entropy_loss(entropy_mode="target"):
    if entropy_mode == "target":
        return target_entropy_loss
    if entropy_mode == "maximization":
        return margin_entropy_loss
    raise ValueError(f"unknown entropy_mode {entropy_mode!r}")
