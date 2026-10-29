"""Generate numerical reference fixtures from the ORIGINAL Keras/TF1 scrambler.

Run in the TF1 env (see README):  .envs/tf1/bin/python reference/make_fixtures.py
Writes tests/fixtures/*.npz, consumed by tests/test_equivalence.py.

Sampling is made deterministic without re-implementing it: `tf.multinomial` is
replaced by the Gumbel-max trick with fixed noise (an exact categorical
sampler), and the uniform draw inside TF's RelaxedOneHotCategorical is replaced
by the same fixed noise. Everything else is the unmodified original code.
"""
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "fixtures")
sys.path.insert(0, os.path.join(HERE, "scrambler"))
for mod in ["isolearn", "isolearn.keras"]:  # imported but unused by the scrambler code
    sys.modules[mod] = types.ModuleType(mod)

import tensorflow as tf  # noqa: E402
import keras  # noqa: E402
from keras import backend as K  # noqa: E402
from keras.layers import Conv1D, Dense, Flatten, Input  # noqa: E402
from keras.models import Model, Sequential, load_model  # noqa: E402
from tensorflow.contrib.distributions.python.ops import relaxed_onehot_categorical as roc  # noqa: E402
from scipy.signal import gaussian  # noqa: E402

import scrambler.models.scrambler_models as sm  # noqa: E402

CASES = {
    # inclusion, straight-through, symmetric sigmoid KL, template + non-uniform background, smoothing, even kernel
    "incl_st_sym": dict(scrambler_mode="inclusion", sample_mode="st", n_classes=1, out_act="sigmoid",
                        nll_mode="reconstruction", predictor_task="classification_sym", reference="predictor",
                        entropy_mode="target", entropy_bits=0.25, entropy_weight=1.0, template=True,
                        zeropad_input=False, mask_dropout=False,
                        net=dict(n_groups=1, n_resblocks_per_group=2, n_channels=8, window_size=8, dilation_rates=[1],
                                 norm_mode="instance", mask_smoothing=True, mask_smoothing_window_size=5,
                                 mask_smoothing_std=1.0)),
    # occlusion, Gumbel, regression MSE, 2 dilated groups
    "occl_gumbel_mse": dict(scrambler_mode="occlusion", sample_mode="gumbel", n_classes=1, out_act="linear",
                            nll_mode="reconstruction", predictor_task="regression", reference="predictor",
                            entropy_mode="target", entropy_bits=1.0, entropy_weight=0.5, template=False,
                            zeropad_input=False, mask_dropout=False,
                            net=dict(n_groups=2, n_resblocks_per_group=1, n_channels=8, window_size=3,
                                     dilation_rates=[1, 2], norm_mode="instance", mask_smoothing=False,
                                     mask_smoothing_window_size=5, mask_smoothing_std=1.0)),
    # inclusion, ST, 3-class softmax KL, batch norm, mask-dropout input (rates 0), zero-padding, margin entropy
    "incl_st_softmax_bn": dict(scrambler_mode="inclusion", sample_mode="st", n_classes=3, out_act="softmax",
                               nll_mode="reconstruction", predictor_task="classification", reference="predictor",
                               entropy_mode="maximization", entropy_bits=0.5, entropy_weight=2.0, template=True,
                               zeropad_input=True, mask_dropout=True,
                               net=dict(n_groups=1, n_resblocks_per_group=2, n_channels=8, window_size=4,
                                        dilation_rates=[2], norm_mode="batch", mask_smoothing=True,
                                        mask_smoothing_window_size=7, mask_smoothing_std=1.5,
                                        mask_drop_scales=[1, 5], mask_min_drop_rate=0.0, mask_max_drop_rate=0.0)),
    # occlusion against binary labels (label inversion), sigmoid KL
    "occl_st_label": dict(scrambler_mode="occlusion", sample_mode="st", n_classes=1, out_act="sigmoid",
                          nll_mode="reconstruction", predictor_task="classification", reference="label",
                          entropy_mode="target", entropy_bits=0.0, entropy_weight=1.0, template=False,
                          zeropad_input=False, mask_dropout=False,
                          net=dict(n_groups=1, n_resblocks_per_group=1, n_channels=8, window_size=5,
                                   dilation_rates=[1], norm_mode="instance", mask_smoothing=False,
                                   mask_smoothing_window_size=5, mask_smoothing_std=1.0)),
}
B, S, L, C = 4, 3, 20, 4
N_STEPS, LR, BETAS = 3, 1e-3, (0.5, 0.9)
NET_DEFAULTS = dict(drop_rate=0.0, mask_drop_scales=[1, 5], mask_min_drop_rate=0.0, mask_max_drop_rate=0.5,
                    label_input=False)


def pin_sampling_noise(u):
    """u: (B, S, L, C) uniform noise in the torch layout -> TF flat order (S, B, L, C)."""
    flat = np.ascontiguousarray(u.transpose(1, 0, 2, 3).reshape(-1, C)).astype(np.float32)
    gumbel = -np.log(-np.log(flat))
    tf.multinomial = lambda logits, n: tf.argmax(logits + tf.constant(gumbel), axis=1)[:, None]
    roc.random_ops = types.SimpleNamespace(random_uniform=lambda shape, **kw: tf.constant(flat))


def tiny_predictor(rng, n_classes, out_act):
    model = Sequential([Conv1D(6, 5, activation="relu", input_shape=(L, C)), Flatten(),
                        Dense(n_classes, activation=out_act)])
    model.set_weights([rng.normal(0, 0.5, w.shape).astype(np.float32) for w in model.get_weights()])
    return model


def layer_weights(layers, prefix):
    return {f"{prefix}/{l.name}/{i}": w for l in layers for i, w in enumerate(l.get_weights())}


def run_case(name, cfg, seed):
    K.clear_session()
    rng = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    x = np.eye(C, dtype=np.float32)[rng.randint(0, C, (B, L))]
    template = np.zeros((L, C), dtype=np.float32)
    if cfg["template"]:
        template[:3] = x[0, :3]
        template[-2:] = np.eye(C)[[1, 2]]
        x[:, :3], x[:, -2:] = template[:3], template[-2:]
    background = rng.dirichlet(np.ones(C) * 2, size=L).astype(np.float32)
    group = np.zeros((B, 1), dtype=np.float32)
    drop = (rng.rand(B, L) > 0.2).astype(np.float32)
    y = rng.randint(0, 2, (B, cfg["n_classes"])).astype(np.float32)
    u = rng.uniform(1e-6, 1 - 1e-6, (B, S, L, C)).astype(np.float32)
    pin_sampling_noise(u)

    net = dict(NET_DEFAULTS, **cfg["net"])
    scr = sm.Scrambler(n_inputs=1, n_classes=cfg["n_classes"], scrambler_mode=cfg["scrambler_mode"], input_size_x=1,
                       input_size_y=L, n_out_channels=C, input_templates=[template[None]],
                       input_backgrounds=[background], batch_size=B, n_samples=S, sample_mode=cfg["sample_mode"],
                       zeropad_input=cfg["zeropad_input"], mask_dropout=cfg["mask_dropout"], network_config=net)
    predictor = tiny_predictor(rng, cfg["n_classes"], cfg["out_act"])
    pred_weights = layer_weights(predictor.layers, "pred")

    # Build the original training graph (n_epochs=0 builds scr.loss_model without fitting).
    x4 = x[:, None]
    scr.train(predictor, x.copy(), y, x.copy(), y, 0, nll_mode=cfg["nll_mode"],
              predictor_task=cfg["predictor_task"], reference=cfg["reference"], entropy_mode=cfg["entropy_mode"],
              entropy_bits=cfg["entropy_bits"], entropy_weight=cfg["entropy_weight"],
              adam_lr=LR, adam_beta_1=BETAS[0], adam_beta_2=BETAS[1])
    lm = scr.loss_model
    scr_layers = [l for l in lm.layers if l.name.startswith("scrambler_") and l.get_weights()]
    out = dict(layer_weights(scr_layers, "w0"), **pred_weights)

    drops = [drop[:, None, :, None]] if cfg["mask_dropout"] else []
    labels = [y] if cfg["reference"] == "label" else []

    # Eval-mode forward of the scrambler model (learning phase 0).
    pwm_logits, pwm, samples, scores = scr.scrambler_model.predict([group, x4] + drops, batch_size=B)
    out.update({"eval/pwm_logits": pwm_logits[:, 0], "eval/pwm": pwm[:, 0], "eval/samples": samples[:, :, 0],
                "eval/scores": scores[:, 0, :, 0]})

    # Train-mode losses and gradients (learning phase 1), as used by fit().
    feeds = lm._feed_inputs + lm._feed_targets + lm._feed_sample_weights + [K.learning_phase()]
    tensors = [lm.get_layer("nll").output, lm.get_layer("entropy").output, lm.total_loss,
               lm.get_layer("t_inflate_non_scrambled_prediction").output,
               lm.get_layer("t_inflate_scrambled_prediction").output]
    weights = lm.trainable_weights
    fn = K.function(feeds, tensors + K.gradients(lm.total_loss, weights))
    inputs = [group, x4] + drops + labels
    dummy, ones = np.zeros((B, 1)), np.ones(B)
    res = fn(inputs + [dummy, dummy] + [ones, ones] + [1])
    out.update({"train/nll": res[0], "train/entropy": res[1], "train/loss": np.float32(res[2]),
                "train/y_ref": res[3], "train/y_scrambled": res[4]})
    out.update({f"grad/{w.name}": g for w, g in zip(weights, res[5:])})

    # A few Adam steps. Keras 2.2.4's compiled Adam is racy under TF1: parameter assignments and the
    # `iterations` increment sit in one unordered update list, so backprop may read already-updated weights
    # and bias correction may use t+1. We apply its exact formula (keras/optimizers.py, Adam.get_updates)
    # sequentially to race-free Keras gradients instead.
    step_fn = K.function(feeds, [lm.total_loss, K.mean(lm.get_layer("nll").output),
                                 cfg["entropy_weight"] * K.mean(lm.get_layer("entropy").output)]
                         + K.gradients(lm.total_loss, weights))
    params = K.batch_get_value(weights)
    m, v, losses = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params], []
    for t in range(1, N_STEPS + 1):
        res_t = step_fn(inputs + [dummy, dummy] + [ones, ones] + [1])
        losses.append(res_t[:3])
        lr_t = LR * np.sqrt(1. - BETAS[1] ** t) / (1. - BETAS[0] ** t)
        for i, g in enumerate(res_t[3:]):
            m[i] = BETAS[0] * m[i] + (1. - BETAS[0]) * g
            v[i] = BETAS[1] * v[i] + (1. - BETAS[1]) * g ** 2
            params[i] = (params[i] - lr_t * m[i] / (np.sqrt(v[i]) + K.epsilon())).astype(np.float32)
        K.batch_set_value(list(zip(weights, params)))
    out["steps/losses"] = np.array(losses, np.float32)
    out.update(layer_weights(scr_layers, "w1"))

    out.update({"x": x, "y": y, "drop": drop, "u": u, "template": template, "background": background})
    config = dict(cfg, net=net, B=B, S=S, L=L, C=C, n_steps=N_STEPS, lr=LR, betas=BETAS,
                  trainable=[w.name for w in weights])
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), config=json.dumps(config), **out)
    print(f"{name}: loss {res[2]:.5f}, {len(weights)} trainable tensors")


def run_per_example_case(seed=0):
    """Score parametrisations of the original per-example optimize/finetune models."""
    K.clear_session()
    rng = np.random.RandomState(seed)
    out = {}
    x = np.eye(C, dtype=np.float32)[rng.randint(0, C, (B, L))][:, None]
    drop = (rng.rand(B, 1, L, 1) > 0.2).astype(np.float32)
    pretrained = rng.uniform(0, 6, (B, 1, L, 1)).astype(np.float32)
    kernel = gaussian(5, 1.0).reshape(1, 5, 1, 1)
    for kind in ["optimize", "finetune"]:
        for mode in ["inclusion", "occlusion"]:
            for norm in ["instance", "conv", None]:
                kw = dict(scrambler_mode=mode, n_out_channels=C, mask_smoothing=True, smooth_window_size=5,
                          norm_mode=norm)
                xi, di, pi = Input(batch_shape=(B, 1, L, C)), Input(batch_shape=(B, 1, L, 1)), Input(batch_shape=(B, 1, L, 1))
                if kind == "optimize":
                    f, pre = sm.load_optimization_model(B, 1, L, **kw), "ot_"
                    model = Model([xi, di], list(f(xi, di)))
                else:
                    f, pre = sm.load_finetuning_model(B, 1, L, max_score_clip=4.0, **kw), "ft_"
                    model = Model([xi, di, pi], list(f(xi, di, pi)))
                for layer in model.layers:
                    if layer.get_weights():
                        new = kernel if layer.name.endswith("smooth_conv") else None
                        new = [new] if new is not None else [rng.normal(0, 1, w.shape).astype(np.float32)
                                                            for w in layer.get_weights()]
                        layer.set_weights(new)
                        out.update({f"{kind}/{mode}/{norm}/{layer.name[len(pre):]}/{i}": w for i, w in enumerate(new)})
                logits, scores = model.predict([x, drop] + ([pretrained] if kind == "finetune" else []), batch_size=B)
                out[f"{kind}/{mode}/{norm}/logits"] = logits[:, 0]
                out[f"{kind}/{mode}/{norm}/scores"] = scores[:, 0, :, 0]
    np.savez_compressed(os.path.join(OUT, "per_example.npz"), x=x[:, 0], drop=drop[:, 0, :, 0],
                        pretrained=pretrained[:, 0, :, 0], **out)
    print("per_example: done")


def run_optimus5(n=64):
    import pandas as pd
    d = os.path.join(HERE, "scrambler", "examples", "dna")
    seqs = pd.read_csv(os.path.join(d, "randomSampleTestingAllAUGtypes.csv"))["utr"].str[:50].tolist()[:n]
    x = np.eye(C, dtype=np.float32)[["ACGT".index(c) for s in seqs for c in s.upper()]].reshape(n, 50, C)
    model = load_model(os.path.join(d, "saved_models", "optimusRetrainedMain.hdf5"))
    np.savez_compressed(os.path.join(OUT, "optimus5.npz"), x=x, y=model.predict(x, batch_size=n))
    print("optimus5: done")


if __name__ == "__main__":
    # Each job runs in a fresh process: TF1/Keras keep global state that otherwise
    # leaks between cases and perturbs the optimizer trajectories.
    import subprocess
    jobs = list(CASES) + ["per_example", "optimus5"]
    if len(sys.argv) > 1:
        job = sys.argv[1]
        if job in CASES:
            run_case(job, CASES[job], seed=list(CASES).index(job))
        else:
            {"per_example": run_per_example_case, "optimus5": run_optimus5}[job]()
    else:
        os.makedirs(OUT, exist_ok=True)
        print("tensorflow", tf.__version__, "keras", keras.__version__)
        for job in jobs:
            subprocess.run([sys.executable, os.path.abspath(__file__), job], check=True)
