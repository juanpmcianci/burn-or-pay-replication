"""T4: (a) hurdles kL, kE across fee schedules and reference positions (psi=1.5, rf=0.20);
(b) robustness of rf=0.20 roots to the lower boundary (x_lo=-9);
(c) gap at interior burn shares and the global burn-vs-pay comparison."""
from rt import *
from multiprocessing import Pool
from scipy.optimize import brentq

def has(p, **kw):
    try: solve2(p, **kw); return True
    except Exception: return False

def kmax(psi, rf, cal, **kw):
    lo, hi = 0.0, 1.5
    if has(make(psi, rf, hi, **cal), **kw): return hi
    for _ in range(12):
        m = 0.5*(lo+hi); lo, hi = (m, hi) if has(make(psi, rf, m, **cal), **kw) else (lo, m)
    return lo

def g(kind, psi, rf, k, cal, x, **kw):
    return gap(make(psi, rf, k, **cal), x, kind, **kw)

def root(kind, psi, rf, km, cal, x, **kw):
    grid = np.linspace(0, 0.97*km, 10); v = []
    for k in grid:
        try: v.append(g(kind, psi, rf, k, cal, x, **kw))
        except Exception: v.append(np.nan)
    for a, b, va, vb in zip(grid[:-1], grid[1:], v[:-1], v[1:]):
        if va < 0 <= vb: return brentq(lambda k: g(kind, psi, rf, k, cal, x, **kw), a, b, xtol=2e-3), v[-1]
    return np.nan, v[-1]

def case(arg):
    lab, psi, rf, cal, kw = arg
    p0 = make(psi, rf, 0.0, **cal); x = x_at_theta(p0)
    km = kmax(psi, rf, cal, **kw)
    kL, eL = root("F", psi, rf, km, cal, x, **kw); kE, eE = root("U", psi, rf, km, cal, x, **kw)
    return f"{lab}: x={x:.2f} kmax={km:.3f} kL={kL:.3f} kE={kE:.3f} kL/kE={kL/kE:.2f} G_F@0.97kmax={eL:.2f}"

def interior(arg):
    psi, rf, k = arg
    p = make(psi, rf, k); x = x_at_theta(p); out = []
    for w0 in [0.0, 0.5, 0.9]:
        q = with_(p, w_B=w0)
        th = theta_of_rho(rho_of(np.array([x]), q), q)[0]
        out.append(f"w={w0}: G_F={gap(q, x, 'F'):.2f} Theta={th:.3f}")
    l0 = logW(solve2(with_(p, w_B=0.0))[0], x)[0]; l1 = logW(solve2(with_(p, w_B=1.0))[0], x)[0]
    out.append(f"logW(burn all)-logW(pay all)={l1-l0:.3f}")
    return f"psi={psi} rf={rf} k={k}: " + " | ".join(out)

if __name__ == "__main__":
    cases = [("base s=0.75", 1.5, 0.20, {}, {}), ("s=0.4", 1.5, 0.20, dict(s=0.4), {}),
             ("s=1.5", 1.5, 0.20, dict(s=1.5), {}), ("s=3", 1.5, 0.20, dict(s=3.0), {}),
             ("Fmax=0.06 (ref at midpoint)", 1.5, 0.20, dict(F_max=0.06), {}),
             ("Fmax=0.24", 1.5, 0.20, dict(F_max=0.24), {}), ("Fmin=0.01", 1.5, 0.20, dict(F_min=0.01), {}),
             ("base xlo=-9", 1.5, 0.20, {}, dict(x_lo=-9.0)), ("psi=11.2 rf=0.20 xlo=-9", 11.2, 0.20, {}, dict(x_lo=-9.0)),
             ("psi=1.1 rf=0.30 xlo=-9", 1.1, 0.30, {}, dict(x_lo=-9.0))]
    with Pool(2) as pool:
        for line in pool.imap(interior, [(1.5, 0.20, 0.0), (1.5, 0.20, 0.3), (1.5, 0.20, 0.5)]): print(line, flush=True)
        for line in pool.imap(case, cases): print(line, flush=True)
