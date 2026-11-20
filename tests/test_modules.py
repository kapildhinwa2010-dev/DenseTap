"""Tests for data / train / budget modules. Run: python tests/test_modules.py"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dense_tap.core import DenseCrossLayerTF
from dense_tap.data import make_toroidal_task, torus_lift, LIFT_DIM, RAW_DIM
from dense_tap.train import Trainer, paired_report
from dense_tap.budget import nparams, solve, PRESET_5M, PRESET_20K


def test_task_shapes_and_chance():
    rng = np.random.default_rng(0)
    o, c, y = make_toroidal_task(256, rng, T_obs=3, K=16)
    assert o.shape == (256, 3, 4) and c.shape == (256, 16, 4) and y.shape == (256,)
    assert y.min() >= 0 and y.max() < 16


def test_label_points_to_true_next():
    """The labelled candidate must be the noiseless true next position."""
    rng = np.random.default_rng(3)
    o, c, y = make_toroidal_task(64, rng, noise=0.0, close=0.9)
    th = np.arctan2(o[:, :, 0], o[:, :, 1])
    dth = np.unwrap(th, axis=1)[:, -1] - np.unwrap(th, axis=1)[:, -2]
    pred_th = th[:, -1] + 5 * dth
    sel = c[np.arange(64), y]
    got = np.arctan2(sel[:, 0], sel[:, 1])
    err = np.abs(((got - pred_th + np.pi) % (2 * np.pi)) - np.pi)
    assert np.median(err) < 0.05, float(np.median(err))


def test_lift_dim_and_unit_norms():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, (8, 3, 4)).astype(np.float32)
    f = torus_lift(z)
    assert f.shape[-1] == LIFT_DIM == 20 and RAW_DIM == 4
    # Only the 16 DERIVED characters are unit-norm. The first 4 entries are the raw
    # noisy observations and are deliberately NOT renormalised.
    derived = f[..., 4:].reshape(*f.shape[:-1], 8, 2)
    norms = np.sqrt((derived ** 2).sum(-1))
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert not np.allclose(np.sqrt((f[..., :2] ** 2).sum(-1)), 1.0, atol=1e-3)


def test_budget_matches_core():
    for cfg in (PRESET_5M, PRESET_20K):
        kw = {k: v for k, v in cfg.items() if k != "nh"}
        m = DenseCrossLayerTF(nh=cfg["nh"], mode="dense", **kw)
        assert m.npar() == nparams(**kw), (m.npar(), nparams(**kw))


def test_solver_hits_5m():
    s = solve(5_000_000)
    assert s is not None and abs(s["params"] - 5_000_000) < 60_000


def test_trainer_reduces_loss():
    rng = np.random.default_rng(0)
    o, c, y = make_toroidal_task(1024, rng)
    ov, cv, yv = make_toroidal_task(256, rng)
    tr = (torus_lift(o), torus_lift(c), y)
    va = (torus_lift(ov), torus_lift(cv), yv)
    net = DenseCrossLayerTF(d=24, nh=4, L=4, din=20, T=3, extra=24, mode="dense")
    t = Trainer(net, tr, va, bs=64, lr=3e-3)
    t.chunk(40, log_every=10)
    first = t.curve[0]["train_loss"]
    t.chunk(200, log_every=10)
    last = t.curve[-1]["train_loss"]
    assert last < first, (first, last)


def test_trainer_respects_time_budget():
    rng = np.random.default_rng(0)
    o, c, y = make_toroidal_task(512, rng)
    tr = (torus_lift(o), torus_lift(c), y)
    net = DenseCrossLayerTF(d=24, nh=4, L=4, din=20, T=3, extra=24, mode="dense")
    t = Trainer(net, tr, None, bs=64)
    t.chunk(100000, max_seconds=0.2)
    assert t.s < 100000 and t.t_total < 3.0


def test_paired_report_math():
    r = paired_report([0.86, 0.85, 0.84], [0.89, 0.88, 0.88])
    assert r["all_seeds_positive"] is True
    assert abs(r["mean_delta"] - np.mean([0.03, 0.03, 0.04])) < 1e-4  # report rounds to 4dp
    assert r["n_seeds"] == 3


def test_taps_change_after_training():
    """Taps must actually move; otherwise they are dead parameters."""
    rng = np.random.default_rng(0)
    o, c, y = make_toroidal_task(1024, rng)
    tr = (torus_lift(o), torus_lift(c), y)
    net = DenseCrossLayerTF(d=24, nh=4, L=4, din=20, T=3, extra=24, mode="dense")
    before = {k: np.array(v) for k, v in net.taps().items()}
    Trainer(net, tr, None, bs=64, lr=3e-3).chunk(150)
    after = {k: np.array(v) for k, v in net.taps().items()}
    moved = max(float(np.abs(after[k] - before[k]).max()) for k in before)
    assert moved > 1e-4, moved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            fails += 1; print("FAIL", fn.__name__, e)
    print(str(len(fns) - fails) + "/" + str(len(fns)) + " passed")
    sys.exit(1 if fails else 0)
