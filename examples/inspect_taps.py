"""P1 mechanism check: where do the taps actually point after training?

If every tap vector concentrates on the immediately-previous layer, dense has
degenerated to seq and any measured gain needs another explanation.

Usage: python examples/inspect_taps.py --steps 1500
"""
import argparse, sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dense_tap import DenseCrossLayerTF, Trainer, make_toroidal_task, torus_lift, PRESET_20K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    o, c, y = make_toroidal_task(16384, rng)
    ov, cv, yv = make_toroidal_task(4096, rng)
    tr = (torus_lift(o), torus_lift(c), y)
    va = (torus_lift(ov), torus_lift(cv), yv)
    cfg = {k: v for k, v in PRESET_20K.items() if k != "nh"}
    net = DenseCrossLayerTF(nh=PRESET_20K["nh"], mode="dense", seed=a.seed, **cfg)
    t = Trainer(net, tr, va, bs=256, lr=3e-3, seed=a.seed)
    t.chunk(a.steps)
    report = {"steps": t.s, "val_acc": t.evaluate(), "layers": {}}
    for l, w in net.taps().items():
        w = np.array(w)
        report["layers"][l] = {
            "weights": [round(float(x), 4) for x in w],
            "mass_on_previous_layer": round(float(w[-1]), 4),
            "mass_on_older_layers": round(float(w[:-1].sum()), 4),
            "entropy_nats": round(float(-(w * np.log(w + 1e-12)).sum()), 4),
            "uniform_entropy": round(float(np.log(len(w))), 4)}
    older = np.mean([v["mass_on_older_layers"] for v in report["layers"].values()])
    report["mean_mass_on_older_layers"] = round(float(older), 4)
    report["interpretation"] = (
        "degenerate-to-seq (taps collapsed onto previous layer)" if older < 0.10
        else "genuine multi-layer read (substantial mass on older layers)")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
