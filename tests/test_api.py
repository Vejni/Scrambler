"""End-to-end smoke tests of the high-level API on a toy motif task."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from scrambler_torch import Scrambler, background_from_data, one_hot_encode

L = 24
NET = dict(n_groups=1, n_resblocks_per_group=1, n_channels=8, window_size=3, dilation_rates=[1], drop_rate=0.0)


class MotifPredictor(nn.Module):
    """P(motif 'GATA' present) via a fixed matching filter."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 1, 4, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor(one_hot_encode("GATA")[0].T[None] * 4.0))

    def forward(self, x):
        return torch.sigmoid(self.conv(x.transpose(1, 2)).amax(-1) - 14.0)


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(0)
    seqs = ["".join(rng.choice(list("ACGT"), L)) for _ in range(64)]
    seqs = [s[:8] + "GATA" + s[12:] if i % 2 == 0 else s for i, s in enumerate(seqs)]
    return one_hot_encode(seqs)


@pytest.mark.parametrize("mode,sample_mode", [("inclusion", "st"), ("occlusion", "gumbel")])
def test_fit_interpret_save_load(tmp_path, data, mode, sample_mode):
    torch.manual_seed(0)
    scr = Scrambler(L, scrambler_mode=mode, input_backgrounds=[background_from_data(data)], n_samples=4,
                    sample_mode=sample_mode, mask_dropout=True, network_config=NET)
    hist = scr.fit(MotifPredictor(), data, None, data[:16], None, n_epochs=2, batch_size=16, entropy_bits=0.5,
                   predictor_task="classification", verbose=False)
    assert len(hist["nll_loss"]) == len(hist["val_nll_loss"]) == 2
    assert np.isfinite(hist["batch_nll_loss"]).all()
    pwm, samples, scores = scr.interpret(data[:5])
    assert pwm.shape == (5, L, 4) and samples.shape == (5, 4, L, 4) and scores.shape == (5, L)
    path = tmp_path / "scr.pt"
    scr.save(path)
    scr2 = Scrambler.load(path)
    x = torch.tensor(data[:5])
    torch.testing.assert_close(scr2.eval().net(x), scr.eval().net(x))


def test_optimize_and_finetune(data):
    torch.manual_seed(0)
    scr = Scrambler(L, network_config=NET, n_samples=4)
    predictor = MotifPredictor()
    for result in [scr.optimize(predictor, data[:6], batch_size=4, n_iters=3, entropy_bits=0.5, verbose=False),
                   scr.finetune(predictor, data[:6], batch_size=4, n_iters=3, entropy_bits=0.5, verbose=False)]:
        pwm, samples, scores, hist = result
        assert pwm.shape == (6, L, 4) and scores.shape == (6, L) and len(hist) == 2
        assert len(hist[0]["nll_loss"]) == 3


def test_one_hot_encode():
    x = one_hot_encode(["ACGTN", "acg"], seq_len=5)
    np.testing.assert_array_equal(x[0].argmax(-1)[:4], [0, 1, 2, 3])
    assert x[0, 4].sum() == 0 and x[1, 3:].sum() == 0 and x[1, :3].sum() == 3


def test_mask_dropout_with_length_not_divisible_by_scale():
    from scrambler_torch.network import mask_dropout
    torch.manual_seed(0)
    m = mask_dropout(torch.ones(3, 1, 23), drop_scales=(1, 5), min_rate=0.5, max_rate=0.5)
    assert m.shape == (3, 1, 23) and set(m.unique().tolist()) <= {0.0, 1.0} and 0 < m.mean() < 1
