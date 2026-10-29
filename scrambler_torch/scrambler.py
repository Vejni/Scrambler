"""High-level Scrambler: training, interpretation and per-example optimization.

PyTorch counterpart of `scrambler.models.Scrambler` (single input, 1D sequences).
The predictor is any frozen nn.Module mapping one-hot (N, L, C) -> (N, K).
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import get_entropy_loss, get_nll_loss
from .network import ScramblerNet, conv1d, glorot_normal_, make_norm, scale_logits, smooth
from .optim import KerasAdam
from .sampler import PSSMSampler, sample_pssm

DEFAULT_NETWORK_CONFIG = dict(n_groups=1, n_resblocks_per_group=4, n_channels=32, window_size=8, dilation_rates=[1],
                              drop_rate=0.25, norm_mode="instance", mask_smoothing=True, mask_smoothing_window_size=7,
                              mask_smoothing_std=1.5, mask_drop_scales=[1, 5], mask_min_drop_rate=0.0,
                              mask_max_drop_rate=0.5)


@dataclass
class ScramblerOutput:
    pwm_logits: torch.Tensor  # (B, L, C)
    pwm: torch.Tensor         # (B, L, C) scrambled PSSM
    samples: torch.Tensor     # (B, S, L, C)
    scores: torch.Tensor      # (B, L) importance scores
    mask: torch.Tensor        # (B, L, C) 1 at free (non-template) positions
    background: torch.Tensor  # (B, L, C)


@dataclass
class Objective:
    """Per-example (nll, entropy) losses; mirrors the loss assembly in the original `train`."""
    scrambler_mode: str = "inclusion"
    nll_mode: str = "reconstruction"
    predictor_task: str = "classification"
    custom_loss_func: object = None
    reference: str = "predictor"  # or 'label'
    entropy_mode: str = "target"
    entropy_bits: float = 0.0
    entropy_weight: float = 1.0

    def __call__(self, predictor, x, y, out):
        b, s = out.samples.shape[:2]
        y_scrambled = predictor(out.samples.flatten(0, 1)).view(b, s, -1)
        coeff = 1.0
        if self.reference == "predictor":
            with torch.no_grad():
                y_ref = predictor(x)
            if self.scrambler_mode == "occlusion":
                coeff = -1.0
        elif self.reference == "label":
            y_ref = y.to(y_scrambled).view(b, -1)
            if self.scrambler_mode == "occlusion" and y_ref.shape[-1] <= 1:
                y_ref = 1.0 - y_ref
        else:
            raise ValueError(f"unknown reference {self.reference!r}")
        if self.custom_loss_func is not None:
            nll_fn, coeff = self.custom_loss_func, 1.0
        else:
            nll_fn = get_nll_loss(self.nll_mode, self.predictor_task, y_scrambled.shape[-1])
        nll = coeff * nll_fn(y_ref[:, None].expand_as(y_scrambled), y_scrambled)
        entropy = get_entropy_loss(self.entropy_mode)(out.pwm, out.mask, out.background, self.entropy_bits)
        return nll, entropy


def _to_tensor(a, dtype=torch.float32):
    return None if a is None else torch.as_tensor(a, dtype=dtype)


def _squeeze_x(x):
    """Accept the original (N, 1, L, C) layout as well as (N, L, C)."""
    return x[:, 0] if x.dim() == 4 and x.shape[1] == 1 else x


def _batches(n, batch_size, shuffle=False):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def _freeze(predictor):
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)
    return predictor


class Scrambler(nn.Module):
    def __init__(self, seq_len, n_out_channels=4, scrambler_mode="inclusion", input_templates=None,
                 input_backgrounds=None, n_samples=32, sample_mode="st", gumbel_temp=0.5, zeropad_input=False,
                 mask_dropout=False, network_config=None):
        super().__init__()
        if input_templates is None:
            input_templates = [np.zeros((seq_len, n_out_channels))]
        if input_backgrounds is None:
            input_backgrounds = [np.full((seq_len, n_out_channels), 1.0 / n_out_channels)]
        input_templates = [np.asarray(t, dtype=np.float32).reshape(seq_len, n_out_channels) for t in input_templates]
        input_backgrounds = [np.asarray(b, dtype=np.float32).reshape(seq_len, n_out_channels) for b in input_backgrounds]
        network_config = {**DEFAULT_NETWORK_CONFIG, **(network_config or {})}
        if network_config.pop("label_input", False):
            raise NotImplementedError("label_input is not ported")

        self.config = dict(seq_len=seq_len, n_out_channels=n_out_channels, scrambler_mode=scrambler_mode,
                           input_templates=input_templates, input_backgrounds=input_backgrounds, n_samples=n_samples,
                           sample_mode=sample_mode, gumbel_temp=gumbel_temp, zeropad_input=zeropad_input,
                           mask_dropout=mask_dropout, network_config=network_config)
        self.scrambler_mode, self.n_samples = scrambler_mode, n_samples
        self.sample_mode, self.gumbel_temp, self.zeropad_input = sample_mode, gumbel_temp, zeropad_input
        self.net = ScramblerNet(n_out_channels, mask_dropout=mask_dropout, **network_config)
        self.sampler = PSSMSampler(input_templates, input_backgrounds)

    @property
    def device(self):
        return self.sampler.mask.device

    def forward(self, x, group=None, drop=None, n_samples=None, uniform_noise=None):
        return self.scramble(x, self.net(x, drop), group, n_samples, uniform_noise)

    def scramble(self, x, scores, group=None, n_samples=None, uniform_noise=None):
        """Turn importance scores into a scrambled PSSM and samples from it."""
        pwm_logits, pwm, mask, background = self.sampler(scale_logits(x, scores, self.scrambler_mode), group)
        samples = sample_pssm(pwm_logits, n_samples or self.n_samples, self.sample_mode, self.gumbel_temp,
                              uniform_noise)
        if self.zeropad_input:
            samples = samples * mask[:, None]
        return ScramblerOutput(pwm_logits, pwm, samples, scores, mask, background)

    # ------------------------------------------------------------------ training
    def fit(self, predictor, x_train, y_train=None, x_val=None, y_val=None, n_epochs=1, batch_size=32,
            group_train=None, group_val=None, adam_lr=1e-4, adam_beta_1=0.5, adam_beta_2=0.9, verbose=True,
            **objective_kwargs):
        """Train the scrambler network against a frozen predictor.

        objective_kwargs: nll_mode, predictor_task, custom_loss_func, reference,
        entropy_mode, entropy_bits, entropy_weight (see `Objective`).
        Returns a history dict of per-batch and per-epoch losses (entropy losses
        unweighted, unlike the Keras history).
        """
        objective = Objective(self.scrambler_mode, **objective_kwargs)
        predictor = _freeze(predictor.to(self.device))
        x_train, y_train = _squeeze_x(_to_tensor(x_train)), _to_tensor(y_train)
        group_train = _to_tensor(group_train, torch.long)
        opt = KerasAdam(self.net.parameters(), lr=adam_lr, betas=(adam_beta_1, adam_beta_2))
        hist = {k: [] for k in ["batch_nll_loss", "batch_entropy_loss", "nll_loss", "entropy_loss",
                                "val_nll_loss", "val_entropy_loss"]}
        for epoch in range(n_epochs):
            self.train()
            for i, idx in enumerate(_batches(len(x_train), batch_size, shuffle=True)):
                nll, ent = self._batch_losses(predictor, objective, x_train, y_train, group_train, idx)
                loss = nll.mean() + objective.entropy_weight * ent.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                hist["batch_nll_loss"].append(nll.mean().item())
                hist["batch_entropy_loss"].append(ent.mean().item())
            n = i + 1
            hist["nll_loss"].append(float(np.mean(hist["batch_nll_loss"][-n:])))
            hist["entropy_loss"].append(float(np.mean(hist["batch_entropy_loss"][-n:])))
            msg = f"epoch {epoch + 1}/{n_epochs}  nll {hist['nll_loss'][-1]:.4f}  entropy {hist['entropy_loss'][-1]:.4f}"
            if x_val is not None:
                v_nll, v_ent = self.evaluate(predictor, x_val, y_val, group_val, batch_size, **objective_kwargs)
                hist["val_nll_loss"].append(v_nll)
                hist["val_entropy_loss"].append(v_ent)
                msg += f"  val_nll {v_nll:.4f}  val_entropy {v_ent:.4f}"
            if verbose:
                print(msg, flush=True)
        return hist

    def _batch_losses(self, predictor, objective, x, y, group, idx):
        dev = self.device
        xb = x[idx].to(dev)
        yb = None if y is None else y[idx].to(dev)
        gb = None if group is None else group[idx].to(dev)
        return objective(predictor, xb, yb, self(xb, gb))

    @torch.no_grad()
    def evaluate(self, predictor, x, y=None, group=None, batch_size=32, **objective_kwargs):
        """Mean (nll, entropy) losses in eval mode."""
        objective = Objective(self.scrambler_mode, **objective_kwargs)
        predictor = _freeze(predictor.to(self.device))
        x, y, group = _squeeze_x(_to_tensor(x)), _to_tensor(y), _to_tensor(group, torch.long)
        self.eval()
        nlls, ents = zip(*(self._batch_losses(predictor, objective, x, y, group, idx)
                           for idx in _batches(len(x), batch_size)))
        return torch.cat(nlls).mean().item(), torch.cat(ents).mean().item()

    # ------------------------------------------------------------ interpretation
    @torch.no_grad()
    def interpret(self, x, group=None, drop=None, batch_size=32, n_samples=None):
        """Returns numpy (pwm (N, L, C), samples (N, S, L, C), scores (N, L))."""
        self.eval()
        x, group, drop = _squeeze_x(_to_tensor(x)), _to_tensor(group, torch.long), _to_tensor(drop)
        outs = []
        for idx in _batches(len(x), batch_size):
            o = self(x[idx].to(self.device), None if group is None else group[idx].to(self.device),
                     None if drop is None else drop[idx].to(self.device), n_samples)
            outs.append((o.pwm.cpu(), o.samples.cpu(), o.scores.cpu()))
        return tuple(torch.cat(t).numpy() for t in zip(*outs))

    # ------------------------------------------------- per-example optimization
    def optimize(self, predictor, x, y=None, batch_size=32, n_iters=500, norm_mode="instance", **kwargs):
        """Optimize importance scores from scratch for each example (no network)."""
        return self._optimize(predictor, x, y, batch_size, n_iters, norm_mode, None, None, **kwargs)

    def finetune(self, predictor, x, y=None, batch_size=32, n_iters=500, norm_mode="instance", max_score_clip=4.0,
                 group=None, drop=None, **kwargs):
        """Refine the network's scores per example: s = relu(clip(S(X), 0, max) - softplus(m))."""
        _, _, pretrained = self.interpret(x, group, drop, batch_size)
        return self._optimize(predictor, x, y, batch_size, n_iters, norm_mode, pretrained, max_score_clip,
                              group=group, drop=drop, **kwargs)

    def _optimize(self, predictor, x, y, batch_size, n_iters, norm_mode, pretrained, max_score_clip, group=None,
                  drop=None, adam_lr=0.01, adam_beta_1=0.5, adam_beta_2=0.9, verbose=True, **objective_kwargs):
        objective = Objective(self.scrambler_mode, **objective_kwargs)
        predictor = _freeze(predictor.to(self.device))
        x, y, group = _squeeze_x(_to_tensor(x)), _to_tensor(y), _to_tensor(group, torch.long)
        drop = torch.ones(x.shape[:2]) if drop is None else _to_tensor(drop).reshape(x.shape[:2])
        pretrained = _to_tensor(pretrained)
        self.eval()
        results, histories = [], []
        for b, idx in enumerate(_batches(len(x), batch_size)):
            if verbose and b % 20 == 0:
                print(f"Optimizing batch {b}...", flush=True)
            dev = self.device
            xb, db = x[idx].to(dev), drop[idx].to(dev)
            yb = None if y is None else y[idx].to(dev)
            gb = None if group is None else group[idx].to(dev)
            pb = None if pretrained is None else pretrained[idx].to(dev)
            scorer = PerExampleScores(batch_size, x.shape[1], norm_mode, self.net.smooth_kernel, max_score_clip).to(dev)
            opt = KerasAdam(scorer.parameters(), lr=adam_lr, betas=(adam_beta_1, adam_beta_2))
            hist = {"nll_loss": [], "entropy_loss": []}
            for _ in range(n_iters):
                nll, ent = objective(predictor, xb, yb, self.scramble(xb, scorer(db, pb), gb))
                loss = nll.sum() + objective.entropy_weight * ent.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                hist["nll_loss"].append(nll.sum().item())
                hist["entropy_loss"].append(ent.sum().item())
            with torch.no_grad():
                o = self.scramble(xb, scorer(db, pb), gb)
            results.append((o.pwm.cpu(), o.samples.cpu(), o.scores.cpu()))
            histories.append(hist)
        pwm, samples, scores = (torch.cat(t).numpy() for t in zip(*results))
        return pwm, samples, scores, histories

    # --------------------------------------------------------------- persistence
    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model


class PerExampleScores(nn.Module):
    """Free per-example score parameters (original `load_optimization_model` /
    `load_finetuning_model`): softplus(norm(m)), optionally subtracted from
    clipped pretrained scores, times the drop mask, then smoothed."""

    def __init__(self, batch_size, seq_len, norm_mode="instance", smooth_kernel=None, max_score_clip=None):
        super().__init__()
        self.mask_logits = nn.Parameter(glorot_normal_(torch.empty(batch_size, seq_len)))
        if norm_mode == "conv":
            self.norm = conv1d(1, 1)
        elif norm_mode == "instance":
            self.norm = make_norm("instance", 1)
        else:
            self.norm = nn.Identity()
        self.smooth_kernel = smooth_kernel
        self.max_score_clip = max_score_clip

    def forward(self, drop, pretrained_scores=None):
        n = drop.shape[0]
        s = F.softplus(self.norm(self.mask_logits[:n, None]))
        if pretrained_scores is not None:
            s = torch.relu(pretrained_scores[:, None].clamp(0.0, self.max_score_clip) - s)
        s = s * drop[:, None]
        if self.smooth_kernel is not None:
            s = smooth(s, self.smooth_kernel)
        return s[:, 0]
