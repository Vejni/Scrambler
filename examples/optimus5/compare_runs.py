"""Compare test-set interpretations of independently trained scramblers
(e.g. the original Keras implementation vs. PyTorch runs with different seeds).

    python examples/optimus5/compare_runs.py examples/optimus5/out/{keras,torch_seed0,torch_seed1}

Every run is scored with the same procedure (32 samples from its PSSM, the
ported Optimus 5' predictor), so implementations are compared like for like.
Keras-vs-torch agreement should be on par with torch-vs-torch (seed) agreement.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.optimus5.predictor import Optimus5  # noqa: E402
from scrambler_torch import background_from_data, sample_pssm  # noqa: E402
from scrambler_torch.losses import mean_conservation  # noqa: E402

DATA = ROOT / "reference" / "scrambler" / "examples" / "dna"


def rowwise_pearson(a, b):
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12)


def topk_overlap(a, b, k=5):
    ta, tb = np.argsort(-a, 1)[:, :k], np.argsort(-b, 1)[:, :k]
    return np.mean([len(set(x) & set(y)) / k for x, y in zip(ta, tb)])


@torch.no_grad()
def score_run(pwm, x, predictor, background, n_samples=32, seed=0):
    torch.manual_seed(seed)
    pwm, x = torch.tensor(pwm), torch.tensor(x)
    samples = sample_pssm(pwm.clamp_min(1e-8).log(), n_samples, "st")
    y = predictor(x)[:, 0]
    y_s = predictor(samples.flatten(0, 1))[:, 0].view(len(x), n_samples)
    bits = mean_conservation(pwm, torch.ones_like(pwm), background.expand_as(pwm))
    return dict(fidelity_r=float(np.corrcoef(y, y_s.mean(1))[0, 1]), mse=float(((y[:, None] - y_s) ** 2).mean()),
                mean_bits=float(bits.mean()))


def main(run_dirs):
    runs = {Path(d).name: np.load(Path(d) / "optimus5_test_interpretations.npz") for d in run_dirs}
    x = next(iter(runs.values()))["x"]
    assert all(np.array_equal(r["x"], x) for r in runs.values()), "runs must interpret the same test set"
    from examples.optimus5.train_optimus5 import load
    background = torch.tensor(background_from_data(load(DATA / "bottom5KIFuAUGTop5KIFuAUG.csv")))
    predictor = Optimus5.from_keras(DATA / "saved_models" / "optimusRetrainedMain.hdf5")

    print(f"{'run':<16}{'fidelity r':>12}{'sample MSE':>12}{'mean bits':>11}")
    for name, r in runs.items():
        s = score_run(r["pwm"], x, predictor, background)
        print(f"{name:<16}{s['fidelity_r']:>12.3f}{s['mse']:>12.4f}{s['mean_bits']:>11.3f}")
    print(f"\n{'pair':<32}{'score r (mean)':>15}{'top-5 overlap':>15}")
    for (a, ra), (b, rb) in itertools.combinations(runs.items(), 2):
        r = rowwise_pearson(ra["scores"], rb["scores"])
        print(f"{a + ' vs ' + b:<32}{np.nanmean(r):>15.3f}{topk_overlap(ra['scores'], rb['scores']):>15.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
