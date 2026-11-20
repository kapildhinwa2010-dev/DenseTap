"""Reproduce the experiments recorded in results/measured.json.

Usage:
    python -m dense_tap.bench --scale 20k --seeds 3 --steps 1500
    python -m dense_tap.bench --scale 5m  --seeds 1 --steps 440 --budget-seconds 120

Runs BOTH seq and dense. Refuses to present a win when any run stopped early.
"""
import argparse, json, time
import numpy as np
from .core import DenseCrossLayerTF
from .data import make_toroidal_task, torus_lift
from .train import Trainer, paired_report
from .budget import PRESET_5M, PRESET_20K


def build_data(seed=7, n_train=16384, n_val=4096, lift=True, **task_kw):
    rng = np.random.default_rng(seed)
    Otr, Ctr, Ytr = make_toroidal_task(n_train, rng, **task_kw)
    Ova, Cva, Yva = make_toroidal_task(n_val, rng, **task_kw)
    if lift:
        return (torus_lift(Otr), torus_lift(Ctr), Ytr), (torus_lift(Ova), torus_lift(Cva), Yva)
    return (Otr, Ctr, Ytr), (Ova, Cva, Yva)


def run_one(mode, cfg, nh, train_data, val_data, steps, bs, lr, seed, budget=None):
    net = DenseCrossLayerTF(seed=seed, mode=mode, nh=nh, **cfg)
    tr = Trainer(net, train_data, val_data, bs=bs, lr=lr, seed=seed)
    t0 = time.time()
    while tr.s < steps:
        left = None if budget is None else max(0.0, budget - (time.time() - t0))
        if left is not None and left <= 0:
            break
        tr.chunk(min(50, steps - tr.s), max_seconds=left)
    return {"mode": mode, "seed": seed, "params": net.npar(), "steps_done": tr.s,
            "val_acc": tr.evaluate(), "wall_seconds": round(time.time() - t0, 1),
            "taps_after_training": net.taps(), "curve": tr.curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["20k", "5m"], default="20k")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--budget-seconds", type=float, default=None)
    ap.add_argument("--no-lift", action="store_true", help="ablate the torus feature map")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    preset = PRESET_20K if a.scale == "20k" else PRESET_5M
    cfg = {k: v for k, v in preset.items() if k != "nh"}
    nh = preset["nh"]
    bs = a.bs if a.bs is not None else (256 if a.scale == "20k" else 32)
    lr = a.lr if a.lr is not None else (3e-3 if a.scale == "20k" else 3e-4)
    lift = not a.no_lift
    cfg["din"] = 20 if lift else 4
    train_data, val_data = build_data(lift=lift)
    out = {"scale": a.scale, "config": cfg, "nh": nh, "bs": bs, "lr": lr,
           "torus_lift": lift, "steps_requested": a.steps, "runs": []}
    for mode in ("seq", "dense"):
        for s in range(a.seeds):
            r = run_one(mode, cfg, nh, train_data, val_data, a.steps, bs, lr, s, a.budget_seconds)
            out["runs"].append(r)
            print(json.dumps({k: r[k] for k in ("mode", "seed", "steps_done", "val_acc")}))
    seq = [r["val_acc"] for r in out["runs"] if r["mode"] == "seq"]
    den = [r["val_acc"] for r in out["runs"] if r["mode"] == "dense"]
    complete = all(r["steps_done"] == a.steps for r in out["runs"])
    out["complete"] = complete
    out["paired"] = paired_report(seq, den) if len(seq) == len(den) and seq else None
    if not complete:
        out["WARNING"] = "INCOMPLETE: some runs stopped early. No comparative claim is supported."
        print(out["WARNING"])
    if out["paired"]:
        print(json.dumps(out["paired"], indent=2))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("wrote", a.out)
    return out


if __name__ == "__main__":
    main()
