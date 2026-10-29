"""Train the ORIGINAL Keras scrambler on Optimus 5' (same config as the original
example notebook) and save test-set interpretations for comparison with the port.

    .envs/tf1/bin/python reference/train_optimus5_keras.py --epochs 10 --out examples/optimus5/out/keras
"""
import argparse
import os
import sys
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "scrambler", "examples", "dna")
sys.path.insert(0, os.path.join(HERE, "scrambler"))
for mod in ["isolearn", "isolearn.keras"]:
    sys.modules[mod] = types.ModuleType(mod)

import tensorflow as tf  # noqa: E402
from keras import backend as K  # noqa: E402
from keras.layers import Input, Lambda  # noqa: E402
from keras.models import Model, load_model  # noqa: E402

import scrambler.models.scrambler_models as sm  # noqa: E402


def load(csv):
    seqs = pd.read_csv(os.path.join(DATA, csv))["utr"].str[:50].str.upper().tolist()
    return np.eye(4, dtype=np.float32)[[["ACGT".index(c) for c in s] for s in seqs]][:, None]  # (N, 1, 50, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "examples", "optimus5", "out", "keras"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.threads,
                                                   inter_op_parallelism_threads=2)))

    x_train, x_test = load("bottom5KIFuAUGTop5KIFuAUG.csv"), load("randomSampleTestingAllAUGtypes.csv")
    # Same predictor wrapper as the original optimus5_utils.load_optimus5_predictor.
    inner = load_model(os.path.join(DATA, "saved_models", "optimusRetrainedMain.hdf5"))
    inp = Input(shape=(1, 50, 4))
    predictor = Model(inp, inner(Lambda(lambda x: x[:, 0, ...])(inp)))

    x_mean = (x_train.sum(axis=(0, 1)) + 1.0) / (x_train.shape[0] + 4.0)
    network_config = dict(n_groups=5, n_resblocks_per_group=4, n_channels=32, window_size=3,
                          dilation_rates=[1, 2, 4, 2, 1], drop_rate=0.0, norm_mode="instance", mask_smoothing=False,
                          mask_smoothing_window_size=5, mask_smoothing_std=1.0, mask_drop_scales=[1, 5],
                          mask_min_drop_rate=0.0, mask_max_drop_rate=0.5, label_input=False)
    scrambler = sm.Scrambler(scrambler_mode="inclusion", input_size_x=1, input_size_y=50, n_out_channels=4,
                             input_templates=[np.zeros((1, 50, 4))], input_backgrounds=[x_mean], batch_size=32,
                             n_samples=32, sample_mode="gumbel", zeropad_input=False, mask_dropout=False,
                             network_config=network_config)
    y_dummy_train, y_dummy_test = np.zeros((len(x_train), 1)), np.zeros((len(x_test), 1))
    history = scrambler.train(predictor, x_train, y_dummy_train, x_test, y_dummy_test, args.epochs,
                              nll_mode="reconstruction", predictor_task="regression", entropy_mode="target",
                              entropy_bits=0.125, entropy_weight=10.0)
    pwm, _, scores = scrambler.interpret(x_test)
    np.savez_compressed(os.path.join(args.out, "optimus5_test_interpretations.npz"), x=x_test[:, 0], pwm=pwm[:, 0],
                        scores=scores[:, 0, :, 0], **{k: np.array(v) for k, v in history.items()})
    print("saved to", args.out)


if __name__ == "__main__":
    main()
