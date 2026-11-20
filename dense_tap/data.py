"""Synthetic toroidal-dynamics task used in every measurement in this repo.

A point moves on T^2 with constant angular velocities. The model sees T_obs noisy
(sin,cos) observations and must pick the correct future position out of K candidates.

THIS IS SYNTHETIC AND TOROIDAL BY CONSTRUCTION. A torus-aware feature map has a
structural advantage here. Results on this task do NOT transfer to language.
"""
import numpy as np

LIFT_DIM = 20
RAW_DIM = 4


def make_toroidal_task(n, rng, T_obs=3, K=16, horizon=5, noise=0.25, close=0.35):
    """Returns obs (n,T_obs,4), cand (n,K,4), labels (n,). Chance accuracy = 1/K."""
    th0 = rng.uniform(0, 2 * np.pi, n)
    ph0 = rng.uniform(0, 2 * np.pi, n)
    w1 = rng.choice(np.array([1, 2, 3, 4, 5]) * np.pi / 6, n)
    w2 = rng.choice(np.array([1, 2, 3, 4]) * np.pi / 6, n)
    t = np.arange(T_obs + horizon)
    TH = th0[:, None] + w1[:, None] * t
    PH = ph0[:, None] + w2[:, None] * t
    obs = np.stack([np.sin(TH[:, :T_obs]), np.cos(TH[:, :T_obs]),
                    np.sin(PH[:, :T_obs]), np.cos(PH[:, :T_obs])], -1)
    obs = obs + rng.normal(0, noise, (n, T_obs, 4))
    tn = np.stack([np.sin(TH[:, -1]), np.cos(TH[:, -1]),
                   np.sin(PH[:, -1]), np.cos(PH[:, -1])], -1)
    cand = [tn]
    for _ in range(K - 1):
        dt = rng.uniform(close, close + .9, n) * rng.choice([-1, 1], n)
        dp = rng.uniform(close, close + .9, n) * rng.choice([-1, 1], n)
        cand.append(np.stack([np.sin(TH[:, -1] + dt), np.cos(TH[:, -1] + dt),
                              np.sin(PH[:, -1] + dp), np.cos(PH[:, -1] + dp)], -1))
    cand = np.stack(cand, 1)
    perm = rng.random((n, K)).argsort(1)
    cand = np.take_along_axis(cand, perm[:, :, None], 1)
    y = (perm == 0).argmax(1)
    return obs.astype(np.float32), cand.astype(np.float32), y


def torus_lift(z):
    """Map (sin th, cos th, sin ph, cos ph) -> 20-dim toroidal characters.

    Base 4 (raw, not renormalised) + harmonics k=2,3 (8) + pairwise phases (8).
    NOTE: fixed feature map; atan2 derivatives are NOT backpropagated.
    This is a documented approximation, not an oversight.
    """
    th = np.arctan2(z[..., 0], z[..., 1])
    ph = np.arctan2(z[..., 2], z[..., 3])
    f = [z]
    for k in (2, 3):
        f.append(np.stack([np.sin(k * th), np.cos(k * th),
                           np.sin(k * ph), np.cos(k * ph)], -1))
    f.append(np.stack([np.sin(th + ph), np.cos(th + ph), np.sin(th - ph), np.cos(th - ph),
                       np.sin(2 * th + ph), np.cos(2 * th + ph),
                       np.sin(th + 2 * ph), np.cos(th + 2 * ph)], -1))
    return np.concatenate(f, -1).astype(np.float32)
