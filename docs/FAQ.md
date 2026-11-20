# FAQ

**Is this novel?**
No. DenseNet-style cross-layer connections pre-exist. This repo is an independent
implementation plus honest measurement inside a transformer. Do not cite it as an invention.

**Why NumPy and not PyTorch?**
The sandbox where this was built had a broken PyTorch install. Hand-writing the backward
pass forced a finite-difference verification, which caught real issues. core.py now serves
as an executable specification for a future PyTorch port.

**Does dense beat seq at 5M params?**
Unknown. The seq baseline at 5M was never run. The README marks that section INCOMPLETE
and CI enforces the label.

**Why did the all-to-all matrix version fail?**
It adds thousands of parameters per layer and sums every layer indiscriminately.
Measured: 57.1% / 13.0% at lr 3e-3, 66.9% at lr 1e-3, versus ~89% for scalar taps.
More connectivity is not the win; learned selective connectivity is.

**What about Shree Yantra geometry?**
Tested four ways (graph topology at two scales, transformer channel mask, three seeds).
No measurable gain over random, dense, or absent wiring. Recorded as a negative result.
Only the mathematical skeleton was used; no spiritual claim was tested, because none is testable.

**Is the torus representation lossless?**
The unit-norm property is exact (verified). But a decaying scan forgets (R2 ~ 0.006 at T=128)
and a unitary scan superimposes tokens instead of storing them separately.
"No information loss" is false as a memory claim.

**Can I use this on GPT-2 today?**
Not out of the box. See ROADMAP P2. The tap mechanism is L(L+1)/2-1 scalars, so a
frozen-base adapter is cheap to build, but it has not been built or tested here.

**Why does the gradient check sample only some entries?**
ReLU kinks make central differences report false failures when the analytic gradient is
near zero. The rule (|grad| > 1e-4) is written in the test file so it cannot be mistaken
for cherry-picking.
