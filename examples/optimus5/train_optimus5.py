"""Train an inclusion scrambler on the Optimus 5' UTR predictor (port of the
original `examples/dna/scrambler_optimus5_example.ipynb`, same hyperparameters).

    python examples/optimus5/train_optimus5.py --epochs 10 --device cuda

Writes the checkpoint, training history and test-set interpretations to --out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.optimus5.predictor import Optimus5  # noqa: E402
from scrambler_torch import Scrambler, background_from_data, one_hot_encode  # noqa: E402

DATA = ROOT / "reference" / "scrambler" / "examples" / "dna"
NETWORK_CONFIG = dict(n_groups=5, n_resblocks_per_group=4, n_channels=32, window_size=3, dilation_rates=[1, 2, 4, 2, 1],
                      drop_rate=0.0, norm_mode="instance", mask_smoothing=False)


def load(csv):
    return one_hot_encode(pd.read_csv(csv)["utr"].str[:50].tolist(), seq_len=50)


@torch.no_grad()
def fidelity(predictor, x, samples, device, batch_size=256):
    """Pearson r between predictions on the originals and the mean prediction over their scrambled samples."""
    y, y_s = [], []
    for i in range(0, len(x), batch_size):
        xb, sb = torch.tensor(x[i:i + batch_size], device=device), torch.tensor(samples[i:i + batch_size], device=device)
        y.append(predictor(xb)[:, 0].cpu())
        y_s.append(predictor(sb.flatten(0, 1)).view(*sb.shape[:2]).mean(1).cpu())
    return float(np.corrcoef(torch.cat(y), torch.cat(y_s))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample-mode", default="gumbel", choices=["gumbel", "st"])
    ap.add_argument("--entropy-bits", type=float, default=0.125)
    ap.add_argument("--entropy-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (torch default if unset)")
    ap.add_argument("--out", type=Path, default=ROOT / "examples" / "optimus5" / "out")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_train = load(DATA / "bottom5KIFuAUGTop5KIFuAUG.csv")
    x_test = load(DATA / "randomSampleTestingAllAUGtypes.csv")
    predictor = Optimus5.from_keras(DATA / "saved_models" / "optimusRetrainedMain.hdf5")

    scrambler = Scrambler(50, scrambler_mode="inclusion", input_backgrounds=[background_from_data(x_train)],
                          n_samples=32, sample_mode=args.sample_mode, network_config=NETWORK_CONFIG).to(args.device)
    history = scrambler.fit(predictor, x_train, None, x_test, None, n_epochs=args.epochs, batch_size=32,
                            nll_mode="reconstruction", predictor_task="regression", entropy_mode="target",
                            entropy_bits=args.entropy_bits, entropy_weight=args.entropy_weight)

    pwm, samples, scores = scrambler.interpret(x_test)
    history["test_fidelity_r"] = fidelity(predictor, x_test, samples, args.device)
    print(f"test fidelity (Pearson r, original vs mean scrambled prediction): {history['test_fidelity_r']:.3f}")
    scrambler.save(args.out / "optimus5_scrambler.pt")
    np.savez_compressed(args.out / "optimus5_test_interpretations.npz", x=x_test, pwm=pwm, scores=scores)
    (args.out / "history.json").write_text(json.dumps(history))


if __name__ == "__main__":
    main()
