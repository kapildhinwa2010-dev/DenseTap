"""Chunked Adam trainer with wall-clock budgeting and resumable state."""
import time
import numpy as np
from .core import softmax


class Trainer:
    def __init__(self, net, train_data, val_data=None, bs=32, lr=3e-4,
                 warmup=50, seed=0, beta1=0.9, beta2=0.999, eps=1e-8):
        self.net = net
        self.Ftr, self.CFtr, self.Ytr = train_data
        self.val_data = val_data
        self.bs, self.lr, self.warmup = bs, lr, warmup
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.r = np.random.default_rng(seed)
        self.m = {n: np.zeros_like(getattr(net, n)) for n in net.names}
        self.v = {n: np.zeros_like(getattr(net, n)) for n in net.names}
        self.s = 0
        self.t_total = 0.0
        self.curve = []

    def chunk(self, nsteps, max_seconds=None, log_every=50):
        """Run up to nsteps updates. Stops early if max_seconds elapsed."""
        t0 = time.time()
        net, bs = self.net, self.bs
        for _ in range(nsteps):
            if max_seconds is not None and time.time() - t0 > max_seconds:
                break
            self.s += 1
            s = self.s
            idx = self.r.integers(0, len(self.Ftr), bs)
            lg, c = net.fwd(self.Ftr[idx], self.CFtr[idx])
            p = softmax(lg)
            yb = self.Ytr[idx]
            loss = float(-np.log(p[np.arange(bs), yb] + 1e-12).mean())
            dl = p.copy()
            dl[np.arange(bs), yb] -= 1
            dl /= bs
            g = net.bwd(c, dl)
            lrt = self.lr * min(1.0, s / max(1, self.warmup))
            for n in net.names:
                self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * g[n]
                self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * g[n] ** 2
                getattr(net, n)[...] -= lrt * (self.m[n] / (1 - self.b1 ** s)) / (
                    np.sqrt(self.v[n] / (1 - self.b2 ** s)) + self.eps)
            if s % log_every == 0:
                self.curve.append({"step": s, "train_loss": round(loss, 4)})
        self.t_total += time.time() - t0
        return {"step": self.s, "wall_seconds": round(self.t_total, 2)}

    def evaluate(self, n=1024):
        if self.val_data is None:
            return None
        Fva, CFva, Yva = self.val_data
        n = min(n, len(Fva))
        lg, _ = self.net.fwd(Fva[:n], CFva[:n])
        return round(float((softmax(lg).argmax(1) == Yva[:n]).mean()), 4)

    def state_dict(self):
        return {"step": self.s, "wall_seconds": self.t_total, "curve": self.curve,
                "params": {n: getattr(self.net, n).tolist() for n in self.net.names},
                "taps": self.net.taps()}


def paired_report(seq_accs, dense_accs):
    """Per-seed deltas, not just means. The honest way to report."""
    seq = np.asarray(seq_accs, float)
    den = np.asarray(dense_accs, float)
    assert seq.shape == den.shape, "need same number of seeds"
    delta = den - seq
    return {"n_seeds": int(len(seq)),
            "seq_mean": round(float(seq.mean()), 4), "seq_std": round(float(seq.std()), 4),
            "dense_mean": round(float(den.mean()), 4), "dense_std": round(float(den.std()), 4),
            "paired_deltas": [round(float(x), 4) for x in delta],
            "mean_delta": round(float(delta.mean()), 4),
            "delta_std": round(float(delta.std()), 4),
            "all_seeds_positive": bool((delta > 0).all()),
            "note": "descriptive only; no significance test claimed at this sample size"}
