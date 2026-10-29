# scrambler-torch

A minimal PyTorch port of **Scrambler networks** (Linder et al., *Nat. Mach. Intell.* 4, 41–54 (2022);
original TF1/Keras code: [johli/scrambler](https://github.com/johli/scrambler)) for interpreting
sequence-predictive models such as DNA/RNA CNNs. The port is checked numerically against the original code.

A scrambler is a ResNet that predicts per-position importance scores *S(X)*. The scores set sampling
temperatures of a PSSM that interpolates between the input sequence and a background distribution. Sequences
sampled from that PSSM are fed to the frozen predictor:

- **Inclusion:** keep the prediction while maximizing PSSM entropy, which finds a minimal sufficient feature set.
- **Occlusion:** destroy the prediction while minimizing entropy.

## Install

```bash
conda env create -p .envs/torch -f environment.yml   # or: pip install -e .
reference/fetch.sh                                    # original repo: example data + Keras predictors
```

## Usage

```python
from scrambler_torch import Scrambler, one_hot_encode, background_from_data, sequence_template

x_train = one_hot_encode(train_seqs)                     # (N, L, 4) float32
predictor = MyTorchModel().eval()                        # one-hot (N, L, 4) -> (N, K)

scrambler = Scrambler(
    seq_len=L, scrambler_mode="inclusion",               # or "occlusion"
    input_templates=[sequence_template("ACG$$$...$")],   # '$' = free position, letters = fixed
    input_backgrounds=[background_from_data(x_train)],
    n_samples=32, sample_mode="st",                      # or "gumbel"
    network_config=dict(n_groups=5, n_resblocks_per_group=4, n_channels=32, window_size=3,
                        dilation_rates=[1, 2, 4, 2, 1], drop_rate=0.0, mask_smoothing=False),
).to("cuda")

history = scrambler.fit(predictor, x_train, x_val=x_test, n_epochs=10, batch_size=32,
                        nll_mode="reconstruction", predictor_task="regression",   # see losses.get_nll_loss
                        entropy_mode="target", entropy_bits=0.125, entropy_weight=10.0)
pwm, samples, scores = scrambler.interpret(x_test)       # (N, L, 4), (N, S, L, 4), (N, L)

# Per-example refinement (paper eq. 16) or optimisation from scratch:
pwm, samples, scores, hist = scrambler.finetune(predictor, x_test[:64], n_iters=500, entropy_bits=0.0)
pwm, samples, scores, hist = scrambler.optimize(predictor, x_test[:64], n_iters=500, entropy_bits=0.125)

scrambler.save("scrambler.pt"); scrambler = Scrambler.load("scrambler.pt")
```

Keras scramblers trained with the original code can be loaded with
`scrambler_torch.convert.load_keras_scrambler_weights(scrambler.net, read_keras_h5("model.h5"))`.

Full example (the original Optimus 5' UTR notebook, same hyperparameters):
`python examples/optimus5/train_optimus5.py --epochs 10` (25–50 min on 4 CPU threads, depending on load).

## Credits and license

MIT. This is a derivative work of [johli/scrambler](https://github.com/johli/scrambler) (© 2021 johli, MIT).
Please cite the paper: J. Linder, A. La Fleur, Z. Chen, A. Ljubetič, D. Baker, S. Kannan, G. Seelig,
"Interpreting neural networks for biological sequences by learning stochastic masks",
*Nat. Mach. Intell.* 4, 41–54 (2022). The ported Optimus 5' predictor loads weights from the original repo
(Sample et al., *Nat. Biotechnol.* 2019); example data and Keras weights are not redistributed here and are
fetched with `reference/fetch.sh`.

## Layout

```
scrambler_torch/
  network.py    ScramblerNet (dilated ResNet -> scores), mask dropout, score -> logit scaling
  sampler.py    templates/background -> PSSM; straight-through and Gumbel sampling
  losses.py     prediction losses (KL, symmetric KL, MSE, max/min) and conservation (entropy) losses
  scrambler.py  Scrambler: fit / evaluate / interpret / optimize / finetune / save / load
  optim.py      KerasAdam (Keras 2.2's exact Adam update rule)
  convert.py    Keras .h5 -> torch weights (h5py only)
  utils.py      one-hot encoding, templates, background
examples/optimus5/   ported Optimus 5' predictor, training script, Keras-vs-torch comparison
reference/           fixture generator + Keras training script (TF1 env), fetch.sh for the original repo
tests/               equivalence tests (+ fixtures), sampler properties, API smoke tests
```

## Mapping to the original

| original (`scrambler.models`)                  | port                                     |
|------------------------------------------------|------------------------------------------|
| `Scrambler(...)`, `network_config` keys        | `Scrambler(seq_len, ...)`, same keys     |
| `Scrambler.train(...)`                         | `Scrambler.fit(...)` (same objective kwargs, `adam_lr`, `adam_beta_*`) |
| `interpret`, `optimize`, `finetune`            | same names; scores are `(N, L)`          |
| `load_scrambler_network`, `make_resblock`      | `network.ScramblerNet`, `network.ResBlock` |
| `build_sampler`, `initialize_templates`, `sample_pwm_st/gumbel` | `sampler.PSSMSampler`, `sampler.sample_pssm` |
| `load_optimization_model`, `load_finetuning_model` | `scrambler.PerExampleScores`          |
| `get_*` loss factories                         | `losses.*`, `losses.get_nll_loss`        |

**Not ported:** 2D (image) inputs, multiple inputs (siamese/joint), `label_input`, extra predictor inputs
(wrap your predictor instead), the training monitor/GIF tooling, and plotting.

**Deliberate differences:**
- Inputs are `(N, L, C)`; the original `(N, 1, L, C)` layout is accepted too.
- Data is not trimmed to a multiple of the batch size, and `n_classes` is inferred from the predictor output.
- The classification `maximization`/`minimization` losses take `(y_true, y_pred)`. In the original they are
  defined with only `y_pred`, while the training code calls every loss with two arguments (found by reading the
  code; we did not run them).
- Mask dropout also works when `L` is not divisible by the drop scale (covered by a test).

## Numerical equivalence with the original

`reference/make_fixtures.py` runs the **unmodified original code** (TF 1.13.1, Keras 2.2.4) and stores small
`.npz` fixtures in `tests/fixtures/`. `tests/test_equivalence.py` loads the Keras weights into the port and
compares the two. Sampling noise is pinned in both implementations without re-implementing it:
`tf.multinomial` is replaced by Gumbel-max with fixed noise (an exact categorical sampler), and the uniform draw
inside TF's `RelaxedOneHotCategorical` is replaced by the same noise. The port samples the same way.

Four configurations cover:
- inclusion/occlusion, ST/Gumbel;
- sigmoid/symmetric/softmax KL, MSE, label reference with binary inversion;
- target/margin entropy, templates, non-uniform background;
- instance/batch norm, even and odd kernels, dilation, smoothing, mask-dropout input, zero-padding.

Limits of the comparison: the random mask dropout is only exercised with drop rates of 0 (its RNG cannot match
TF's), so it is checked for shape and behaviour but not numerically against the original. The `optimize` and
`finetune` loops are smoke-tested end to end; only their score parametrisation is compared numerically.

For each configuration the tests check:

| check                                                               | tolerance                       |
|---------------------------------------------------------------------|---------------------------------|
| eval forward: scores, PSSM logits, PSSM, samples                     | rtol 1e-4, atol 1e-5            |
| train-mode per-example losses, total loss                            | rtol 1e-4, atol 1e-5            |
| gradients of every trainable tensor                                  | 1e-3 rel. to the tensor's scale |
| 3 Adam steps: losses at every step; final weights                    | 1e-4; ≥99.5% within 2e-5, all < lr |
| per-example `optimize`/`finetune` score models (all norm modes)       | rtol 1e-4, atol 1e-5            |
| ported Optimus 5' predictor vs. Keras model                          | 1e-4                            |

Things found along the way (each handled in the tests or the port):
- **Keras 2.2.4's compiled Adam is nondeterministic under TF1.** Replaying the same training steps from
  identical weights and optimizer state sometimes gives different losses (e.g. 5 of 7 replays agreed). The
  likely cause is that parameter updates and the `iterations` increment sit in one unordered TF update list,
  but we did not verify the mechanism. The fixtures therefore apply Keras's exact Adam formula
  (`keras/optimizers.py`) sequentially to Keras's gradients.
- **Keras puts Adam's ε outside the bias correction.** `torch.optim.Adam` puts it inside. This changes steps
  for gradient entries around 1e-5, so the port ships `KerasAdam` and uses it by default.
- **Inert parameters.** The resblock `conv_0` bias feeds straight into a normalization layer. Its gradient is
  analytically zero (float noise in both frameworks), so it is excluded from the weight comparison.
- **Ill-conditioned elements.** Adam's first step is about `lr·sign(g)`, so the few elements with near-zero
  gradient are sensitive to float32 noise. A torch run with 1e-7-perturbed weights deviates just as much. This
  is why the final-weight check is distributional.
- **Keras reports entropy weighted.** The entropy loss in Keras's training logs is already multiplied by
  `entropy_weight`.

### Training-level check (Optimus 5', 10 epochs, original notebook settings)

The original Keras scrambler and two torch seeds were trained with identical settings and scored with the same
procedure on the 3,200 test UTRs (`examples/optimus5/compare_runs.py`):

| run         | val nll | fidelity r¹ | mean bits |
|-------------|--------:|------------:|----------:|
| Keras       | 0.553   | 0.839       | 0.189     |
| torch seed 0| 0.563   | 0.841       | 0.181     |
| torch seed 1| 0.558   | 0.838       | 0.176     |

| pair                 | importance-score r² | top-5 position overlap |
|----------------------|--------------------:|-----------------------:|
| Keras vs torch seed 0| 0.866               | 0.748                  |
| Keras vs torch seed 1| 0.868               | 0.752                  |
| torch seed 0 vs 1    | 0.848               | 0.738                  |

¹ Pearson r between the predictor output on each sequence and its mean over 32 samples from the scrambled PSSM.
² Mean per-sequence Pearson r.

Keras-vs-torch agreement is as high as seed-to-seed agreement within torch. Keras's epoch-averaged *training*
nll was somewhat higher (0.50 vs 0.42–0.45), but validation losses match.

## Running things

```bash
CUDA_VISIBLE_DEVICES= .envs/torch/bin/python -m pytest -q                  # ~5 s, CPU
CUDA_VISIBLE_DEVICES= .envs/tf1/bin/python reference/make_fixtures.py       # regenerate fixtures (~1 min)
# Optimus 5' (CPU 25-50 min each with 4 threads; add --device cuda on a free GPU):
.envs/torch/bin/python examples/optimus5/train_optimus5.py --epochs 10 --seed 0 --out examples/optimus5/out/torch_seed0
.envs/tf1/bin/python reference/train_optimus5_keras.py --epochs 10 --out examples/optimus5/out/keras   # ~50 min
.envs/torch/bin/python examples/optimus5/compare_runs.py examples/optimus5/out/{keras,torch_seed0,torch_seed1}
```
