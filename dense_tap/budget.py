"""Budget solver: pick (d, L, extra) to hit a target parameter count."""


def nparams(d, L, extra, din=20, T=3, dense=True):
    p = 8 * d * d * L + din * d + T * d + 2 * d * extra + extra * din
    if dense:
        p += L * (L + 1) // 2 - 1
    return p


def solve(target=5_000_000, din=20, T=3, Ls=(4, 6, 8), extras=(64, 128, 256),
          d_range=range(64, 400, 4), tolerance=60_000):
    best = None
    for L in Ls:
        for extra in extras:
            for d in d_range:
                p = nparams(d, L, extra, din, T, True)
                if abs(p - target) < tolerance:
                    if best is None or abs(p - target) < abs(best["params"] - target):
                        best = {"d": int(d), "nh": 8 if d % 8 == 0 else 4, "L": L,
                                "extra": extra, "din": din, "T": T, "params": p}
    return best


PRESET_5M = {"d": 320, "nh": 8, "L": 6, "din": 20, "T": 3, "extra": 128}
PRESET_20K = {"d": 24, "nh": 4, "L": 4, "din": 20, "T": 3, "extra": 24}
