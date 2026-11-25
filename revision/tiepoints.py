"""Revision A1: lifecycle tie points under the feasible derivative, against the uniform benchmark.
For each case: kappa_max (largest feedback with a price, bisection), then a grid scan of G on [0, 0.995 kappa_max]
and the first sign change from - to + refined with brentq. The gap is evaluated at the state with Theta = 0.45."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import brentq
from feasible import *

NG = 24

def make(psi, rf, k, **kw):
    return calibrated(beta=psi, kappa=k, F_max=0.12, F_min=0.0, r_f=rf, **kw)

def gap(kind, psi, rf, k, **kw):
    p = make(psi, rf, k, **kw)
    x = x_at_theta(p)
    return (G_feasible if kind == "F" else G_uniform)(p, x)[0]

def kmax(psi, rf, **kw):
    lo, hi = 0.0, 2.0
    if has_price(make(psi, rf, hi, **kw)):
        return hi
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if has_price(make(psi, rf, mid, **kw)) else (lo, mid)
    return lo

def first_root(kind, psi, rf, km, **kw):
    grid = np.linspace(0, 0.995 * km, NG)
    vals = []
    for k in grid:
        try: vals.append(gap(kind, psi, rf, k, **kw))
        except Exception: vals.append(np.nan)
    vals = np.array(vals)
    for a, b, va, vb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(va) and np.isfinite(vb) and va < 0 <= vb:
            return brentq(lambda k: gap(kind, psi, rf, k, **kw), a, b, xtol=1e-4), vals
    return np.nan, vals

def case(args):
    psi, rf, kw = args
    if not has_price(make(psi, rf, 0.0, **kw)):
        return dict(psi=psi, rf=rf, **kw, kmax=np.nan, kF=np.nan, kU=np.nan, note="no price at kappa=0")
    km = kmax(psi, rf, **kw)
    kF, vF = first_root("F", psi, rf, km, **kw)
    kU, vU = first_root("U", psi, rf, km, **kw)
    return dict(psi=psi, rf=rf, **kw, kmax=km, kF=kF, kU=kU, GF_edge=vF[-1], GU_edge=vU[-1],
                GF_max=np.nanmax(vF), note="")

if __name__ == "__main__":
    cases = [(psi, rf, {}) for psi in [1.1, 1.5, 2.0] for rf in [0.06, 0.10, 0.15, 0.20, 0.30]]
    with Pool(8) as pool:
        res = pool.map(case, cases)
    d = pd.DataFrame(res)
    d.to_csv("output/tiepoints.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.round(4).to_string(index=False))
