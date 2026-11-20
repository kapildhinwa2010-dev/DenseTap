# DENSE-TAP

**Every layer reads every previous layer - for 20 extra scalars.**

[![ci](https://github.com/Kapildhinwa2010-dev/DenseTap/actions/workflows/ci.yml/badge.svg)](https://github.com/Kapildhinwa2010-dev/DenseTap/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-17%2F17%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![deps](https://img.shields.io/badge/deps-numpy%20only-lightgrey)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/5M%20benchmark-INCOMPLETE-orange)
![novelty](https://img.shields.io/badge/novelty%20claim-none-informational)

`+2.72 points` · `20 extra scalars` · `3/3 seeds positive` · `0.0004% param overhead`

> **Honesty contract:** every number here was measured. Unrun experiments are labelled
> INCOMPLETE and CI fails if that label is removed.

**Contents:** [Idea](#the-idea) · [Verified](#verified) · [Measured](#measured) · [NOT claimed](#not-claimed) · [Quickstart](#quickstart) · [Docs](#docs) · [Status](#status)

> Idea, direction and system design: **Kapil Dhinwa** ([@Kapildhinwa2010-dev](https://github.com/Kapildhinwa2010-dev))
> Implementation co-built with AI. Every number here was measured in a sandbox; nothing is estimated.

## The idea

A standard transformer is a chain: layer 7 sees layer 6, layer 6 sees layer 5.
Information from layer 2 reaches layer 7 only by surviving five transformations.

DENSE-TAP makes it a weighted DAG:

```
standard (seq):   h_l = Block(h_{l-1})
DENSE-TAP:        x_l = sum_j alpha_lj * h_j   (j <= l),  alpha_l = softmax(AL_l)
                  h_l = Block(x_l)
```

`AL_l` holds `l+1` scalars. Whole-model overhead: **L(L+1)/2 - 1 scalars**.
At L=6 that is **20 numbers** on a 5,007,040-param model - **0.0004%**.

Init at zero -> uniform softmax -> starts as a plain average of all previous layers,
then learns where to look. `model.taps()` exposes the learned weights per layer.

### Why it is a plugin
Only the taps need training. Freeze the base model, train `L(L+1)/2 - 1` scalars,
and you have changed its cross-layer information flow. Cheap enough to test honestly.

## Verified

| Check | Result |
|---|---|
| Unit tests | **8/8 pass** |
| Gradient check `seq` | worst rel. err **7.28e-05** |
| Gradient check `dense` | worst rel. err **4.26e-05** |
| Reduction invariant | dense with one-hot-last taps == seq |
| Param overhead | dense - seq = **exactly 20** scalars at L=6 |
| 5M budget | **5,007,060** params (d=320, nh=8, L=6, extra=128) |
| Finite outputs | no NaN/Inf |

Gradients hand-derived, checked vs central finite differences in float64.
Tests sample entries with non-tiny analytic gradient to avoid ReLU-kink 0/0 artifacts;
that sampling rule is written in the test, not hidden.

## Measured

### 20K params - COMPLETE, this is the real result
Synthetic hard toroidal next-point selection. K=16, chance = 6.25%.
d=24, nh=4, L=4, 1500 steps, bs=256, lr=3e-3. Identical data and schedule for both.

| Wiring | Params | seed 0 | seed 1 | seed 2 | Mean | Std |
|---|---|---|---|---|---|---|
| **dense** | 20,625 | 90.23% | 89.16% | 88.43% | **89.27%** | 0.74 |
| seq | 20,616 | 88.35% | 86.30% | 85.01% | 86.55% | 1.38 |

**Paired per-seed deltas: +1.88, +2.86, +3.42 points. 3/3 positive. Mean +2.72, sigma 0.64.**
Seed variance also dropped (1.38 -> 0.74). Overhead: 9 scalars.

### 5M params - INCOMPLETE
`dense`, d=320/nh=8/L=6, bs=32, lr=3e-4, 440 steps, 74.5 s wall on 2 CPU cores:

| Step | Train loss | Val acc |
|---|---|---|
| 50 | 2.662 | - |
| 150 | 1.265 | 80.27% |
| 250 | 0.772 | 84.67% |
| 350 | 0.715 | 86.52% |
| 440 | 0.700 | 87.99% |

Trains stably, no collapse. **But the `seq` baseline at 5M was never run.**
No comparative claim at 5M scale is supported by this repo. This is a stub, not a result.

Sandbox: 2.04 GB RAM (2048 MB cgroup cap), 2 cores, OpenBLAS 2 threads,
~222 GFLOP/s on 512x512 matmul, ~140 ms/step at bs=32, weights+grads+Adam ~76 MB.
PyTorch unavailable, so everything is hand-written NumPy.

## NOT claimed

Read this before quoting anything.

- **Not novel.** DenseNet-style cross-layer connections pre-exist. This is an independent
  implementation and measurement in a transformer, not an invention.
- **Not tested on language.** One synthetic toroidal task. Zero real text, zero pretrained LLMs.
- **Not tested at depth or length.** L=4, T=3 tokens. Real transformers are 12-96 layers.
- **Not shown at 5M.** Baseline missing.
- **Mechanism unexplained.** Taps were never inspected after training. We do not know
  whether the gain comes from learned routing or just from averaging.
- **Bigger is not better:** 5M at 440 steps (88.0%) was *behind* 20K at 1500 steps (89.3%).
  Parameter count without training budget buys nothing.

### Sibling negative results (kept, not buried)

<details>
<summary><b>Expand: four falsified hypotheses</b></summary>
Same experimental arc. Recorded so nobody re-runs them blindly:

| Hypothesis | Verdict |
|---|---|
| Shree-Yantra geometric wiring beats random/dense/none | NO - 4 configs, delta<1pt, within seed noise sigma 1.3-1.5 |
| Literal all-to-all cross-layer **matrix** (not scalar taps) | NO - unstable: 57.1% / 13.0% @ lr 3e-3; 66.9% @ lr 1e-3 |
| Torus has no information loss | NO - decaying scan R2~0.006 @ T=128; unitary variant preserves norm exactly but superimposes (token-0 R2 0.185 -> 0.027) |
| Torus feature lifting helps on cyclic data | YES - +31.8 pts, replicated on 2 architectures at matched ~10K params |

</details>

**Lesson from the all-to-all failure:** more connections is not the win.
*Learned, selective* connections is the win. A full cross-layer matrix dilutes the signal
and breaks optimisation; 20 scalars that learn where to look do not.


## Quickstart

```bash
git clone https://github.com/Kapildhinwa2010-dev/DenseTap.git
cd DenseTap
pip install numpy
python tests/test_core.py       # 8/8
python tests/test_modules.py    # 9/9
python examples/quickstart.py   # 60-second demo (NOT the benchmark)
```

Reproduce the actual headline result:

```bash
python -m dense_tap.bench --scale 20k --seeds 3 --steps 1500
```

Ablate the torus feature map (the control behind the +31.8 point claim):

```bash
python -m dense_tap.bench --scale 20k --seeds 3 --steps 1500 --no-lift
```

Check the mechanism (P1 - never run at full length yet):

```bash
python examples/inspect_taps.py --steps 1500
```

<details>
<summary><b>What the tap inspector tells you</b></summary>

It prints per-layer weights, mass on older layers, and entropy.
If `mean_mass_on_older_layers < 0.10`, the taps collapsed onto the previous layer,
dense has degenerated to seq, and the +2.72 points needs a different explanation.
This is the single most important unrun experiment in the repo.

</details>

## Docs

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | The equation, param accounting, tap gradient path, known approximations |
| [REPRODUCE.md](docs/REPRODUCE.md) | Exact commands for every experiment, sandbox profile |
| [ROADMAP.md](docs/ROADMAP.md) | P0-P3, ordered by value-per-hour, plus known traps |
| [FAQ.md](docs/FAQ.md) | Novelty, why NumPy, why all-to-all failed, Yantra, losslessness |
| [measured.json](results/measured.json) | Every number, machine-readable, with unsupported_claims |
| [CONTRIBUTING.md](CONTRIBUTING.md) | The one rule: no claim without a measurement |
| [CHANGELOG.md](CHANGELOG.md) | 0.1.0 -> 0.2.0 |

## Citing this honestly

If you reference this work, please carry the caveats with it:

```
Dhinwa, K. (2026). DENSE-TAP: dense cross-layer wiring for transformers.
Independent implementation and measurement. +2.72 points over sequential wiring
at 20K parameters on a synthetic toroidal task (3 seeds, all positive).
Not novel as a technique; not validated on language data; 5M comparison incomplete.
https://github.com/Kapildhinwa2010-dev/DenseTap
```

## Layout

```
DenseTap/
  dense_tap/
    core.py               # DenseCrossLayerTF: fwd + hand-derived bwd, seq|dense
    data.py               # toroidal task generator + torus_lift feature map
    train.py              # chunked resumable Adam trainer, paired_report
    budget.py             # parameter-budget solver + presets
    bench.py              # CLI: runs BOTH modes, flags INCOMPLETE runs
  tests/
    test_core.py          # 8 tests: gradients, reduction invariant, budget, taps
    test_modules.py       # 9 tests: labels, lift norms, trainer, tap movement
  examples/
    quickstart.py         # 60-second demo
    inspect_taps.py       # P1 mechanism check
  results/measured.json   # every number + explicit unsupported_claims list
  docs/                   # ARCHITECTURE / REPRODUCE / ROADMAP / FAQ
  .github/
    workflows/ci.yml      # 2 Python versions, both test files, honesty gate
    honesty_gate.py       # fails CI if README drifts from measured.json
    ISSUE_TEMPLATE/       # claim audit, negative result
  pyproject.toml          # pip-installable
```

## Run

```
python tests/test_core.py     # 8/8 should pass; numpy is the only dependency
```

```
from dense_tap import DenseCrossLayerTF
m = DenseCrossLayerTF(d=320, nh=8, L=6, din=20, T=3, extra=128, mode='dense')
print(m.npar())        # 5,007,060
print(m.tap_count())   # 20
print(m.taps())        # learned mixing weights per layer
```

## Status

**Idea preserved. Implementation verified. Benchmarks partially run.**
The 5M comparison and any real-model test are deliberately left open - see docs/ROADMAP.md.
An unfinished experiment is labelled unfinished here, on purpose.

> Learn. Build. Measure. Share. Repeat.
