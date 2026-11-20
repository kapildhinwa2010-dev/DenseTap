# ROADMAP - when time permits

Nothing here is done. Ordered by value-per-hour.

## P0 - close the open loop (the one thing that matters)
- [ ] Run mode='seq' at exactly the 5M config (d=320, nh=8, L=6, extra=128), same seed,
      same bs=32, lr=3e-4, warmup 50, same step count as the dense run.
- [ ] Extend both to >=1500 steps. Dense stopped at 440 due to sandbox time limits.
- [ ] 3 seeds each. Report paired per-seed deltas, not just means.
- [ ] Until this exists, the 5M section of the README stays marked INCOMPLETE.

## P1 - does it survive depth, and WHY does it work?
- [ ] L in {4, 8, 12, 24} at fixed d. Plot delta-vs-depth.
      Hypothesis worth killing: gain may vanish once the residual stream already carries everything.
- [ ] Inspect model.taps() after training. Do taps concentrate on the immediately-previous layer
      (dense degenerates to seq, gain should vanish) or spread across layers?
      This is the mechanism check. Without it, +2.72pts is an unexplained number.

## P2 - leave the synthetic sandbox
- [ ] Port taps to PyTorch (sandbox had it broken; core.py is the reference spec).
- [ ] Frozen GPT-2 + trainable taps only. Train L(L+1)/2 - 1 scalars, nothing else.
      Baseline: frozen GPT-2 with no taps. Any real LM eval.
- [ ] If taps-only training gives no gain on real text, say so and archive. Valid outcome.

## P3 - controls that could kill the result
- [ ] Fixed uniform averaging (no learning) vs learned taps. If uniform matches learned,
      the story is 'averaging stabilises', not 'learned routing helps'.
- [ ] Give seq the same param budget as extra FFN width instead of taps.
- [ ] Longer sequences (T >= 64). T=3 is trivially short.

## Known traps
- All-to-all cross-layer MATRICES collapse. Use scalar taps. Already measured; do not retry blindly.
- ReLU kinks make naive finite-difference checks report false failures. Sample non-tiny-gradient entries.
- Sandbox tool calls cap at ~20 s. Chunk training and persist optimizer state across calls.
- Sandbox has 2 GB RAM / 2 cores. 5M params + Adam fits in ~76 MB; activations dominate.
