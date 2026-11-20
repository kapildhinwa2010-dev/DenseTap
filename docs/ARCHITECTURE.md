# Architecture

## The one equation
```
seq   :  x_l = h_{l-1}
dense :  x_l = sum_{j=0..l} alpha_lj * h_j ,   alpha_l = softmax(AL_l)
h_l   = Block_l(x_l)
```
`Block_l` is unchanged: pre-LN -> causal MHSA -> residual -> pre-LN -> ReLU FFN -> residual.
Only the *input* to each block changes.

## Parameter accounting
| Component | Count |
|---|---|
| per layer (Wq,Wk,Wv,Wo,F1,F2) | 8 d^2 |
| E | din * d |
| PE | T * d |
| U | 2d * extra |
| Pred | extra * din |
| **taps** | **L(L+1)/2 - 1** |

At d=320, L=6, extra=128, din=20, T=3: 5,007,040 (seq) vs 5,007,060 (dense).

## Gradient path through the taps
Two terms, both implemented and finite-difference verified:
```
d h_j   += alpha_lj * d x_l          (route gradient back to every source layer)
d AL_l  += softmax_jacobian(alpha_l) * <h_j, d x_l>
```
The second term makes routing learnable. Without it the taps are dead parameters;
tests/test_modules.py::test_taps_change_after_training guards against that regression.

## Initialisation
AL_l = zeros -> softmax gives uniform 1/(l+1). Training starts from plain averaging,
not from a random preference. Any observed gain is therefore not an artefact of a lucky
initial wiring.

## Reduction invariant
Set AL_l to a one-hot on the last index and dense becomes seq exactly.
Asserted in tests/test_core.py. This is the strongest correctness check in the repo:
it proves the dense path is a strict generalisation, not a different model.

## Known approximations
1. torus_lift is a fixed feature map; atan2 derivatives are not backpropagated.
2. LayerNorm has no learnable scale/shift (simplification, identical in both modes).
3. Readout is mean-pool concat last-token -> MLP -> bilinear candidate score.
   Mean-pool destroys order information; this is why a modular-arithmetic task failed
   early in the project and was replaced.
