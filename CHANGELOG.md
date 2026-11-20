# Changelog

## 0.2.0
- Added dense_tap/data.py: toroidal task generator + torus_lift feature map.
- Added dense_tap/train.py: chunked resumable Adam trainer, paired_report.
- Added dense_tap/budget.py: parameter-budget solver and presets.
- Added dense_tap/bench.py: CLI running BOTH seq and dense; marks early-stopped runs INCOMPLETE.
- Added tests/test_modules.py (9 tests) incl. label-correctness and tap-movement checks.
- Added examples/quickstart.py and examples/inspect_taps.py (P1 mechanism check).
- Added pyproject.toml, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, issue/PR templates.
- Added docs/ARCHITECTURE.md, docs/REPRODUCE.md, docs/FAQ.md.
- CI now runs both test files plus a machine-checked honesty gate on Python 3.10 and 3.12.

## 0.1.0
- Initial reference implementation: DenseCrossLayerTF (seq | dense), manual backprop.
- 8 gradient/invariant tests. results/measured.json with explicit unsupported_claims.
- 20K experiment complete; 5M experiment INCOMPLETE (seq baseline not run).
