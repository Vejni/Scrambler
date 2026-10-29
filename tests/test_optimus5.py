"""The ported Optimus 5' predictor reproduces the Keras model's predictions."""
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.optimus5.predictor import Optimus5

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / "reference" / "scrambler" / "examples" / "dna" / "saved_models" / "optimusRetrainedMain.hdf5"
FIX = ROOT / "tests" / "fixtures" / "optimus5.npz"


@pytest.mark.skipif(not (H5.exists() and FIX.exists()), reason="needs reference repo and fixture")
def test_optimus5_matches_keras():
    fx = np.load(FIX)
    with torch.no_grad():
        y = Optimus5.from_keras(H5)(torch.tensor(fx["x"]))
    np.testing.assert_allclose(y, fx["y"], rtol=1e-4, atol=1e-4)
