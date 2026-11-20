"""60-second quickstart. python examples/quickstart.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dense_tap import DenseCrossLayerTF, Trainer, make_toroidal_task, torus_lift, PRESET_20K

rng = np.random.default_rng(7)
o, c, y = make_toroidal_task(4096, rng)
ov, cv, yv = make_toroidal_task(1024, rng)
tr = (torus_lift(o), torus_lift(c), y)
va = (torus_lift(ov), torus_lift(cv), yv)

cfg = {k: v for k, v in PRESET_20K.items() if k != "nh"}
for mode in ("seq", "dense"):
    net = DenseCrossLayerTF(nh=PRESET_20K["nh"], mode=mode, seed=0, **cfg)
    t = Trainer(net, tr, va, bs=256, lr=3e-3)
    t.chunk(400)
    print(mode, "params", net.npar(), "val_acc", t.evaluate())
    if mode == "dense":
        print("  taps:", {k: [round(x, 3) for x in v] for k, v in net.taps().items()})

print()
print("NOTE: 400 steps is a demo, not the 1500-step benchmark in results/measured.json.")
print("Do not quote these numbers as the repo result.")
