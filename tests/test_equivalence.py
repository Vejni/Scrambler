"""Numerical equivalence with the original Keras/TF1 implementation.

Fixtures are produced by reference/make_fixtures.py (TF1 env) and contain the
original code's weights, pinned sampling noise, outputs, losses, gradients and
the weights after a few Adam steps.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from scrambler_torch import Objective, Scrambler, scale_logits
from scrambler_torch.convert import conv_kernel, keras_tensors, load_keras_scrambler_weights, load_keras_weights, \
    scrambler_layer_map
from scrambler_torch.network import gaussian_kernel
from scrambler_torch.optim import KerasAdam
from scrambler_torch.scrambler import PerExampleScores

FIX = Path(__file__).parent / "fixtures"
CASES = sorted(p.stem for p in FIX.glob("*.npz") if p.stem not in {"per_example", "optimus5"})
TOL = dict(rtol=1e-4, atol=1e-5)

pytestmark = pytest.mark.skipif(not CASES, reason="no fixtures; run reference/make_fixtures.py in the TF1 env")


def layer_dict(fx, prefix):
    """{layer: [arrays]} from fixture keys 'prefix/layer/i'."""
    out = defaultdict(dict)
    for k in fx.files:
        rest = k[len(prefix) + 1:].split("/") if k.startswith(prefix + "/") else []
        if len(rest) == 2:
            out[rest[0]][int(rest[1])] = fx[k]
    return {layer: [ws[i] for i in sorted(ws)] for layer, ws in out.items()}


class TinyPredictor(nn.Module):
    """Mirror of the Keras test predictor: Conv1D(6, 5, valid, relu) -> Flatten -> Dense(K, act)."""

    def __init__(self, weights, act):
        super().__init__()
        conv_w = next(w for k, w in weights.items() if k.startswith("conv1d"))
        dense_w = next(w for k, w in weights.items() if k.startswith("dense"))
        self.conv = nn.Conv1d(4, 6, 5)
        self.dense = nn.Linear(dense_w[0].shape[0], dense_w[0].shape[1])
        load_keras_weights(self.conv, conv_w)
        load_keras_weights(self.dense, dense_w)
        self.act = {"sigmoid": torch.sigmoid, "softmax": lambda z: torch.softmax(z, -1), "linear": lambda z: z}[act]

    def forward(self, x):
        h = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2).flatten(1)
        return self.act(self.dense(h))


def setup(name):
    fx = np.load(FIX / f"{name}.npz")
    cfg = json.loads(str(fx["config"]))
    net_cfg = {k: v for k, v in cfg["net"].items() if k != "label_input"}
    scr = Scrambler(cfg["L"], cfg["C"], cfg["scrambler_mode"], [fx["template"]], [fx["background"]], cfg["S"],
                    cfg["sample_mode"], zeropad_input=cfg["zeropad_input"], mask_dropout=cfg["mask_dropout"],
                    network_config=net_cfg)
    load_keras_scrambler_weights(scr.net, layer_dict(fx, "w0"))
    predictor = TinyPredictor(layer_dict(fx, "pred"), cfg["out_act"]).requires_grad_(False)
    objective = Objective(cfg["scrambler_mode"], cfg["nll_mode"], cfg["predictor_task"], None, cfg["reference"],
                          cfg["entropy_mode"], cfg["entropy_bits"], cfg["entropy_weight"])
    t = {k: torch.tensor(fx[k]) for k in ["x", "y", "drop", "u"]}
    t["drop"] = t["drop"] if cfg["mask_dropout"] else None
    return fx, cfg, scr, predictor, objective, t


def losses(scr, predictor, objective, t):
    out = scr(t["x"], drop=t["drop"], uniform_noise=t["u"])
    nll, ent = objective(predictor, t["x"], t["y"], out)
    return out, nll, ent, nll.mean() + objective.entropy_weight * ent.mean()


def keras_grad_to_torch(scr, keras_name):
    layer, kind = keras_name.split("/")[0], keras_name.split("/")[1].split(":")[0]
    module = scrambler_layer_map(scr.net)[layer]
    index = {"kernel": 0, "bias": 1, "beta": 0, "gamma": 1}[kind]
    if isinstance(module, nn.BatchNorm1d):
        index = {"gamma": 0, "beta": 1}[kind]
    return keras_tensors(module)[index]


def inert(keras_name, cfg):
    """Resblock conv_0 biases feed straight into a normalization layer: they cannot affect the output and
    their gradient is analytically zero (float noise in both frameworks, which Adam then amplifies)."""
    layer, kind = keras_name.split("/")[:2]
    return (cfg["net"]["norm_mode"] in ("instance", "batch") and layer.startswith("scrambler_resblock_")
            and layer.endswith("_conv_0") and kind.startswith("bias"))


@pytest.mark.parametrize("name", CASES)
def test_eval_forward(name):
    fx, cfg, scr, _, _, t = setup(name)
    scr.eval()
    with torch.no_grad():
        out = scr(t["x"], drop=t["drop"], uniform_noise=t["u"])
    np.testing.assert_allclose(out.scores, fx["eval/scores"], **TOL)
    np.testing.assert_allclose(out.pwm_logits, fx["eval/pwm_logits"], **TOL)
    np.testing.assert_allclose(out.pwm, fx["eval/pwm"], **TOL)
    np.testing.assert_allclose(out.samples, fx["eval/samples"], **TOL)


@pytest.mark.parametrize("name", CASES)
def test_train_losses_and_gradients(name):
    fx, cfg, scr, predictor, objective, t = setup(name)
    scr.train()
    _, nll, ent, loss = losses(scr, predictor, objective, t)
    np.testing.assert_allclose(nll.detach(), fx["train/nll"], **TOL)
    np.testing.assert_allclose(ent.detach(), fx["train/entropy"], **TOL)
    np.testing.assert_allclose(loss.item(), fx["train/loss"], **TOL)
    loss.backward()
    grads = [k for k in fx.files if k.startswith("grad/")]
    assert len(grads) == len(cfg["trainable"]) == sum(1 for p in scr.parameters() if p.requires_grad)
    global_scale = max(np.abs(fx[k]).max() for k in grads)
    for k in grads:
        keras_name = k[len("grad/"):]
        param = keras_grad_to_torch(scr, keras_name)
        if inert(keras_name, cfg):
            assert np.abs(fx[k]).max() < 1e-5 * global_scale and param.grad.abs().max() < 1e-5 * global_scale
            continue
        g = conv_kernel(fx[k]) if param.dim() == 3 else fx[k]
        scale = np.abs(fx[k]).max()  # compare relative to the tensor's gradient scale
        np.testing.assert_allclose(param.grad / scale, np.asarray(g) / scale, rtol=1e-3, atol=1e-4,
                                   err_msg=keras_name)


@pytest.mark.parametrize("name", CASES)
def test_adam_steps(name):
    fx, cfg, scr, predictor, objective, t = setup(name)
    scr.train()
    opt = KerasAdam(scr.net.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    for step in range(cfg["n_steps"]):
        _, nll, ent, loss = losses(scr, predictor, objective, t)
        # Fixture stores [total, nll, entropy_weight * entropy] before each update.
        reported = [loss.item(), nll.mean().item(), objective.entropy_weight * ent.mean().item()]
        np.testing.assert_allclose(reported, fx["steps/losses"][step], rtol=1e-4, atol=1e-5)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Adam's first step is ~lr * sign(g), so the few elements with near-zero gradients are sensitive to
    # float32 noise (torch vs. a 1e-7-perturbed torch run differs just as much). Require almost all
    # elements to match tightly and none to differ by more than one learning-rate step.
    final, diffs = layer_dict(fx, "w1"), []
    for layer, module in scrambler_layer_map(scr.net).items():
        for tensor, w, kind in zip(keras_tensors(module), final[layer], ["kernel", "bias"]):  # skips BN stats
            if not inert(f"{layer}/{kind}", cfg):
                w = conv_kernel(w) if tensor.dim() == 3 else torch.as_tensor(w)
                diffs.append((tensor.detach() - w).abs().flatten())
    diffs = torch.cat(diffs)
    assert (diffs < 2e-5).float().mean() > 0.995, f"{(diffs >= 2e-5).sum()} of {diffs.numel()} elements differ"
    assert diffs.max() < cfg["lr"]


@pytest.mark.skipif(not (FIX / "per_example.npz").exists(), reason="no per_example fixture")
@pytest.mark.parametrize("kind", ["optimize", "finetune"])
@pytest.mark.parametrize("mode", ["inclusion", "occlusion"])
@pytest.mark.parametrize("norm", ["instance", "conv", None])
def test_per_example_scores(kind, mode, norm):
    fx = np.load(FIX / "per_example.npz")
    key = f"{kind}/{mode}/{norm}"
    w = layer_dict(fx, key)
    x, drop, pretrained = (torch.tensor(fx[k]) for k in ["x", "drop", "pretrained"])
    kernel = gaussian_kernel(5, 1.0)
    np.testing.assert_allclose(kernel.flatten(), w["scrambler_smooth_conv"][0].flatten(), rtol=1e-6)
    scorer = PerExampleScores(x.shape[0], x.shape[1], norm, kernel, 4.0 if kind == "finetune" else None)
    with torch.no_grad():
        scorer.mask_logits.copy_(torch.tensor(w["scrambler_mask_dense"][0]))
    if norm is not None:
        load_keras_weights(scorer.norm, w["scrambler_mask_norm" if norm == "instance" else "scrambler_mask_conv"])
    with torch.no_grad():
        scores = scorer(drop, pretrained if kind == "finetune" else None)
    np.testing.assert_allclose(scores, fx[key + "/scores"], **TOL)
    np.testing.assert_allclose(scale_logits(x, scores, mode), fx[key + "/logits"], **TOL)
