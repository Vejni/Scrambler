"""Scrambled-PSSM construction and differentiable sampling.

Mirrors `build_sampler` / `initialize_templates` of the original code. Each
template group g holds:
  template_logits[g] : +10 / -4 at fixed template letters, 0 elsewhere
  mask[g]            : 1 at free positions ('$'), 0 at fixed template letters
  log_background[g]  : log B~, the per-position background distribution
and pwm_logits = (scaled_logits + log B~) * mask + template_logits.
"""
import torch
import torch.nn as nn


def template_logits_and_mask(template):
    """template: (L, C) one-hot with all-zero rows at free positions."""
    template = torch.as_tensor(template, dtype=torch.float32)
    fixed = template.sum(-1, keepdim=True) >= 1.0
    logits = torch.where(fixed, torch.where(template.argmax(-1, keepdim=True) == torch.arange(template.shape[-1]),
                                            10.0, -4.0), torch.zeros(()))
    mask = (template.sum(-1, keepdim=True) <= 0.0).float().expand_as(template).clone()
    return logits, mask


class PSSMSampler(nn.Module):
    """Holds templates/backgrounds (frozen buffers) and builds/samples scrambled PSSMs."""

    def __init__(self, templates, backgrounds):
        super().__init__()
        logits, masks = zip(*(template_logits_and_mask(t) for t in templates))
        backgrounds = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in backgrounds])
        self.register_buffer("template_logits", torch.stack(logits))
        self.register_buffer("mask", torch.stack(masks))
        self.register_buffer("background", backgrounds)
        self.register_buffer("log_background", backgrounds.log())

    def forward(self, scaled_logits, group=None):
        """Returns pwm_logits, pwm, mask, background, each (B, L, C)."""
        g = torch.zeros(scaled_logits.shape[0], dtype=torch.long, device=scaled_logits.device) if group is None else group
        mask = self.mask[g]
        pwm_logits = (scaled_logits + self.log_background[g]) * mask + self.template_logits[g]
        return pwm_logits, torch.softmax(pwm_logits, dim=-1), mask, self.background[g]


class _StraightThrough(torch.autograd.Function):
    """Forward: the hard one-hot sample. Backward: identity onto the soft probabilities."""

    @staticmethod
    def forward(ctx, probs, onehot):
        return onehot.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def gumbel_noise(u):
    return -torch.log(-torch.log(u))


def sample_pssm(pwm_logits, n_samples, sample_mode="st", temperature=0.5, uniform_noise=None):
    """Draw n_samples per PSSM: (B, L, C) -> (B, n_samples, L, C).

    'st'     : exact one-hot categorical samples (Gumbel-max) with softmax
               straight-through gradients (original `st_sampled_softmax`).
    'gumbel' : relaxed one-hot (Concrete) samples at `temperature`.
    `uniform_noise` (B, n_samples, L, C) in (0, 1) may be passed to make
    sampling deterministic; otherwise it is drawn with torch.rand.
    """
    logits = pwm_logits[:, None].expand(-1, n_samples, -1, -1)
    if uniform_noise is None:
        tiny = torch.finfo(logits.dtype).tiny
        uniform_noise = torch.rand_like(logits).clamp_(min=tiny)
    g = gumbel_noise(uniform_noise.to(logits))
    if sample_mode == "st":
        probs = torch.softmax(logits, dim=-1)
        onehot = nn.functional.one_hot((logits + g).argmax(-1), logits.shape[-1]).to(logits)
        return _StraightThrough.apply(probs, onehot)
    if sample_mode == "gumbel":
        return torch.softmax((logits + g) / temperature, dim=-1)
    raise ValueError(f"unknown sample_mode {sample_mode!r}")
