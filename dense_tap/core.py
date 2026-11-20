"""DENSE-TAP: dense cross-layer wiring for transformers (pure NumPy reference).

Idea (Kapil Dhinwa): in a standard transformer layer l reads only layer l-1.
Here every layer reads a LEARNED WEIGHTED COMBINATION of ALL previous layers:

    x_l = sum_{j<=l} alpha_lj * h_j ,  alpha_l = softmax(AL_l)   (l+1 learnable scalars)

Overhead: L(L+1)/2 - 1 scalars total (20 at L=6). Nothing else changes.
Gradients verified vs finite differences. See results/measured.json for what is
and is NOT measured. This file makes no performance claim.
"""
import numpy as np


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ln_f(x):
    mu = x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True) + 1e-5
    return (x - mu) / sd, mu, sd


def ln_b(dy, x, mu, sd):
    n = x.shape[-1]
    xh = (x - mu) / sd
    return (1.0 / (n * sd)) * (n * dy - dy.sum(-1, keepdims=True) - xh * (dy * xh).sum(-1, keepdims=True))


class DenseCrossLayerTF:
    """mode=seq (standard chain) or mode=dense (all-layer learned reads).

    Inputs are precomputed features f (B,T,din) and candidates cf (B,K,din).
    logits = <pred, cf> / sqrt(din). Task-agnostic. Manual fwd + bwd, no autodiff.
    """

    def __init__(self, d=32, nh=4, L=4, din=20, T=3, extra=24, seed=0, mode="dense"):
        assert d % nh == 0, "d must be divisible by nh"
        assert mode in ("seq", "dense")
        r = np.random.default_rng(seed)
        self.d, self.nh, self.dk, self.L, self.T, self.mode = d, nh, d // nh, L, T, mode
        sc = 1.0 / np.sqrt(d)
        self.E = r.normal(0, .4, (din, d)).astype(np.float32)
        self.PE = r.normal(0, .2, (T, d)).astype(np.float32)
        self.names = ["E", "PE", "U", "Pred"]
        for l in range(L):
            for nm, sh in [("Wq", (d, d)), ("Wk", (d, d)), ("Wv", (d, d)),
                           ("Wo", (d, d)), ("F1", (d, 2 * d)), ("F2", (2 * d, d))]:
                setattr(self, nm + str(l), r.normal(0, sc, sh).astype(np.float32))
                self.names.append(nm + str(l))
            if mode == "dense" and l > 0:
                setattr(self, "AL" + str(l), np.zeros(l + 1, np.float32))
                self.names.append("AL" + str(l))
        rr = np.random.default_rng(seed + 99)
        self.U = rr.normal(0, 1 / np.sqrt(2 * d), (2 * d, extra)).astype(np.float32)
        self.Pred = rr.normal(0, 1 / np.sqrt(extra), (extra, din)).astype(np.float32)

    def npar(self):
        return sum(getattr(self, n).size for n in self.names)

    def tap_count(self):
        return self.L * (self.L + 1) // 2 - 1 if self.mode == "dense" else 0

    def taps(self):
        if self.mode != "dense":
            return {}
        return {l: softmax(getattr(self, "AL" + str(l))).tolist() for l in range(1, self.L)}

    def fwd(self, f, cf):
        B, T, _ = f.shape
        d, nh, dk = self.d, self.nh, self.dk
        hs = [f @ self.E + self.PE[None]]
        cc = []
        tmask = np.triu(np.ones((T, T), np.float32), 1) * -1e9
        for l in range(self.L):
            if l == 0 or self.mode == "seq":
                x = hs[-1]; ci = None
            else:
                a = softmax(getattr(self, "AL" + str(l)))
                St = np.stack(hs[:l + 1], 0)
                x = np.einsum("j,jbtd->btd", a, St); ci = (a, St)
            xn, mu, sd = ln_f(x)
            q = (xn @ getattr(self, "Wq" + str(l))).reshape(B, T, nh, dk).transpose(0, 2, 1, 3)
            k = (xn @ getattr(self, "Wk" + str(l))).reshape(B, T, nh, dk).transpose(0, 2, 1, 3)
            v = (xn @ getattr(self, "Wv" + str(l))).reshape(B, T, nh, dk).transpose(0, 2, 1, 3)
            at = softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk) + tmask)
            ao = (at @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            h1 = x + ao @ getattr(self, "Wo" + str(l))
            hn, mu2, sd2 = ln_f(h1)
            pre = np.maximum(hn @ getattr(self, "F1" + str(l)), 0)
            h2 = h1 + pre @ getattr(self, "F2" + str(l))
            hs.append(h2)
            cc.append((x, ci, mu, sd, xn, q, k, v, at, ao, h1, mu2, sd2, hn, pre))
        agg = np.concatenate([hs[-1].mean(1), hs[-1][:, -1]], -1)
        u = np.tanh(agg @ self.U)
        pred = u @ self.Pred
        lg = np.einsum("bd,bkd->bk", pred, cf) / np.sqrt(cf.shape[-1])
        return lg, (f, cc, agg, u, pred, cf)

    def bwd(self, cache, dl):
        f, cc, agg, u, pred, cf = cache
        B, T, _ = f.shape
        d, nh, dk = self.d, self.nh, self.dk
        g = {n: np.zeros_like(getattr(self, n)) for n in self.names}
        dp = np.einsum("bk,bkd->bd", dl, cf) / np.sqrt(cf.shape[-1])
        g["Pred"] = u.T @ dp
        du = (dp @ self.Pred.T) * (1 - u ** 2)
        g["U"] = agg.T @ du
        da = du @ self.U.T
        dh = [np.zeros((B, T, d), np.float32) for _ in range(self.L + 1)]
        dh[self.L][:, -1] += da[:, d:]
        dh[self.L] += da[:, None, :d] / T
        for l in range(self.L - 1, -1, -1):
            x, ci, mu, sd, xn, q, k, v, at, ao, h1, mu2, sd2, hn, pre = cc[l]
            dout = dh[l + 1]
            F1 = getattr(self, "F1" + str(l)); F2 = getattr(self, "F2" + str(l))
            g["F2" + str(l)] = pre.reshape(-1, 2 * d).T @ dout.reshape(-1, d)
            dpre = (dout @ F2.T) * (pre > 0)
            g["F1" + str(l)] = hn.reshape(-1, d).T @ dpre.reshape(-1, 2 * d)
            dh1 = dout + ln_b(dpre @ F1.T, h1, mu2, sd2)
            dao = dh1 @ getattr(self, "Wo" + str(l)).T
            g["Wo" + str(l)] = ao.reshape(-1, d).T @ dh1.reshape(-1, d)
            dav = dao.reshape(B, T, nh, dk).transpose(0, 2, 1, 3)
            dat = dav @ v.transpose(0, 1, 3, 2)
            dv = at.transpose(0, 1, 3, 2) @ dav
            dsc = at * (dat - (dat * at).sum(-1, keepdims=True))
            dq = dsc @ k / np.sqrt(dk)
            dkk = dsc.transpose(0, 1, 3, 2) @ q / np.sqrt(dk)
            g["Wq" + str(l)] = xn.reshape(-1, d).T @ dq.transpose(0, 2, 1, 3).reshape(-1, d)
            g["Wk" + str(l)] = xn.reshape(-1, d).T @ dkk.transpose(0, 2, 1, 3).reshape(-1, d)
            g["Wv" + str(l)] = xn.reshape(-1, d).T @ dv.transpose(0, 2, 1, 3).reshape(-1, d)
            dxn = (dq.transpose(0, 2, 1, 3).reshape(B, T, d) @ getattr(self, "Wq" + str(l)).T
                   + dkk.transpose(0, 2, 1, 3).reshape(B, T, d) @ getattr(self, "Wk" + str(l)).T
                   + dv.transpose(0, 2, 1, 3).reshape(B, T, d) @ getattr(self, "Wv" + str(l)).T)
            dx = dh1 + ln_b(dxn, x, mu, sd)
            if ci is None:
                dh[l] += dx
            else:
                a, St = ci
                for j in range(l + 1):
                    dh[j] += a[j] * dx
                da_ = np.einsum("jbtd,btd->j", St, dx)
                g["AL" + str(l)] += a * (da_ - (da_ * a).sum())
        g["E"] = f.reshape(-1, f.shape[-1]).T @ dh[0].reshape(-1, d)
        g["PE"] = dh[0].sum(0)
        return g
