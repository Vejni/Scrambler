"""Properties of the PSSM construction and samplers (no reference implementation needed)."""
import numpy as np
import torch

from scrambler_torch import PSSMSampler, sample_pssm, sequence_template
from scrambler_torch.losses import mean_conservation


def test_template_positions_are_fixed():
    template = sequence_template("AC$$$$G")
    sampler = PSSMSampler([template], [np.full((7, 4), 0.25)])
    logits = torch.randn(2, 7, 4) * 5
    pwm_logits, pwm, mask, _ = sampler(logits)
    assert torch.equal(mask[0, :, 0], torch.tensor([0, 0, 1, 1, 1, 1, 0.0]))
    np.testing.assert_allclose(pwm[:, [0, 1, 6]].argmax(-1), [[0, 1, 2]] * 2)
    assert (pwm[:, [0, 1, 6]].amax(-1) > 0.999).all()
    samples = sample_pssm(pwm_logits, 64, "st")
    assert (samples[:, :, [0, 1, 6]].argmax(-1) == torch.tensor([0, 1, 2])).all()


def test_zero_scores_give_background():
    """Inclusion with S(X) = 0 everywhere: the PSSM equals the background (zero conservation)."""
    bg = torch.distributions.Dirichlet(torch.ones(4)).sample((10,))
    sampler = PSSMSampler([np.zeros((10, 4))], [bg.numpy()])
    _, pwm, mask, background = sampler(torch.zeros(3, 10, 4))
    torch.testing.assert_close(pwm, bg.expand(3, -1, -1))
    torch.testing.assert_close(mean_conservation(pwm, mask, background), torch.zeros(3), atol=1e-6, rtol=0)


def test_st_samples_are_onehot_with_softmax_frequencies():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 4)
    samples = sample_pssm(logits, 20000, "st")
    assert torch.equal(samples.sum(-1), torch.ones(2, 20000, 5))
    assert set(samples.unique().tolist()) == {0.0, 1.0}
    torch.testing.assert_close(samples.mean(1), torch.softmax(logits, -1), atol=0.015, rtol=0)


def test_st_gradient_is_softmax_gradient():
    torch.manual_seed(0)
    logits = torch.randn(3, 6, 4, requires_grad=True)
    w = torch.randn(3, 2, 6, 4)
    (sample_pssm(logits, 2, "st") * w).sum().backward()
    ref = logits.detach().requires_grad_()
    (torch.softmax(ref, -1)[:, None] * w).sum().backward()
    torch.testing.assert_close(logits.grad, ref.grad)


def test_gumbel_samples_are_relaxed_and_concentrate_at_low_temperature():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 4)
    soft = sample_pssm(logits, 10, "gumbel", temperature=0.5)
    torch.testing.assert_close(soft.sum(-1), torch.ones(2, 10, 5))
    u = torch.rand(2, 10, 5, 4)
    cold = sample_pssm(logits, 10, "gumbel", temperature=1e-2, uniform_noise=u)
    hard = sample_pssm(logits, 10, "st", uniform_noise=u)
    assert torch.equal(cold.argmax(-1), hard.argmax(-1))
    assert (cold.amax(-1) > 0.5).all() and cold.amax(-1).median() > 0.99
