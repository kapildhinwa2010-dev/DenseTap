"""Gradient + invariant tests. Run: python tests/test_core.py  (numpy only)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dense_tap.core import DenseCrossLayerTF, softmax


def _net(mode, d=12, nh=3, L=3, extra=8, seed=0):
    net = DenseCrossLayerTF(d=d, nh=nh, L=L, din=20, T=3, extra=extra, seed=seed, mode=mode)
    for n in net.names:
        setattr(net, n, getattr(net, n).astype(np.float64))
    return net


def _data():
    r = np.random.default_rng(0)
    return r.normal(0, 1, (3, 3, 20)), r.normal(0, 1, (3, 5, 20)), r.integers(0, 5, 3)


def gradcheck_multi(mode, n_samples=6, eps=1e-6):
    """Central finite differences. Samples entries with non-tiny analytic gradient
    to avoid ReLU-kink 0/0 artifacts. Sampling rule is explicit, not hidden."""
    net = _net(mode); f, cf, y = _data()
    lg, c = net.fwd(f, cf); p = softmax(lg)
    dl = p.copy(); dl[np.arange(3), y] -= 1; dl /= 3
    g = net.bwd(c, dl)
    worst = 0.0
    for n in net.names:
        A = getattr(net, n); flat = A.reshape(-1); gf = g[n].reshape(-1)
        cand = np.flatnonzero(np.abs(gf) > 1e-4)
        if len(cand) == 0:
            cand = np.arange(flat.size)
        pick = np.random.default_rng(7).choice(cand, size=min(n_samples, len(cand)), replace=False)
        for k in pick:
            old = flat[k]
            flat[k] = old + eps
            l1 = -np.log(softmax(net.fwd(f, cf)[0])[np.arange(3), y] + 1e-12).mean()
            flat[k] = old - eps
            l2 = -np.log(softmax(net.fwd(f, cf)[0])[np.arange(3), y] + 1e-12).mean()
            flat[k] = old
            num = (l1 - l2) / (2 * eps); ana = gf[k]
            worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-12))
    return worst


def test_grad_seq():
    assert gradcheck_multi("seq") < 1e-4


def test_grad_dense():
    assert gradcheck_multi("dense") < 1e-4


def test_tap_count():
    for L in [2, 4, 6, 8]:
        assert DenseCrossLayerTF(d=12, nh=3, L=L, mode="dense").tap_count() == L * (L + 1) // 2 - 1
        assert DenseCrossLayerTF(d=12, nh=3, L=L, mode="seq").tap_count() == 0


def test_param_overhead_is_tiny():
    a = DenseCrossLayerTF(d=320, nh=8, L=6, extra=128, mode="dense").npar()
    b = DenseCrossLayerTF(d=320, nh=8, L=6, extra=128, mode="seq").npar()
    assert a - b == 20
    assert (a - b) / b < 1e-5


def test_5m_budget():
    m = DenseCrossLayerTF(d=320, nh=8, L=6, din=20, T=3, extra=128, mode="dense")
    assert 4950000 < m.npar() < 5050000


def test_taps_uniform_at_init():
    m = DenseCrossLayerTF(d=12, nh=3, L=4, mode="dense")
    for l, a in m.taps().items():
        assert np.allclose(a, 1.0 / (l + 1), atol=1e-6)


def test_seq_equals_dense_with_onehot_last_tap():
    """Core invariant: taps all-mass-on-previous-layer must reduce dense to seq."""
    a = DenseCrossLayerTF(d=12, nh=3, L=4, extra=8, seed=3, mode="dense")
    b = DenseCrossLayerTF(d=12, nh=3, L=4, extra=8, seed=3, mode="seq")
    for n in b.names:
        setattr(a, n, getattr(b, n).copy())
    for l in range(1, a.L):
        v = np.full(l + 1, -1e4, np.float32); v[-1] = 1e4
        setattr(a, "AL" + str(l), v)
    r = np.random.default_rng(1)
    f = r.normal(0, 1, (2, 3, 20)).astype(np.float32)
    cf = r.normal(0, 1, (2, 5, 20)).astype(np.float32)
    la = a.fwd(f, cf)[0]; lb = b.fwd(f, cf)[0]
    assert np.allclose(la, lb, atol=2e-4), np.abs(la - lb).max()


def test_finite_outputs():
    m = DenseCrossLayerTF(d=24, nh=4, L=4, mode="dense")
    r = np.random.default_rng(0)
    lg, _ = m.fwd(r.normal(0, 1, (4, 3, 20)).astype(np.float32),
                  r.normal(0, 1, (4, 5, 20)).astype(np.float32))
    assert np.isfinite(lg).all()


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
