# Contributing

This repo has one rule that outranks all others: **do not add a claim without a measurement.**

## Before you open a PR
1. `python tests/test_core.py` and `python tests/test_modules.py` must both pass.
2. If you change the math, re-run the finite-difference gradient check.
   Note the ReLU-kink trap: sample entries with non-tiny analytic gradient.
3. If you change a number in the README, change it in `results/measured.json` too.
   CI asserts they stay consistent.

## Reporting a result
- Always run the **matched baseline**: same params, data, schedule, seeds.
- Report **paired per-seed deltas**, not just means.
- If a run stopped early, label it INCOMPLETE. `bench.py` does this automatically.
- Negative results are welcome and get their own section. Four hypotheses were falsified
  in this project and that is treated as progress.

## Things already known to fail (do not retry blindly)
- All-to-all cross-layer **matrices** collapse. Scalar taps work; matrices did not.
- Shree-Yantra geometric wiring showed no gain across 4 configurations.
- A decaying selective scan is not lossless, regardless of the addressing scheme.

## Style
- NumPy only in the core. No autodiff dependency in `dense_tap/core.py`.
- Every approximation gets a comment saying it is an approximation.
