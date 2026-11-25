"""UQ level 1 with the feasible derivative (UQ_PLAN amendment D5)."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "uq"))
from uq1 import params                                   # noqa: E402
from feasible import G_feasible, G_uniform, has_price    # noqa: E402

X = np.array([0.0])
def gap(kind, row, k):
    p = params(row, k)
    return (G_feasible if kind == "F" else G_uniform)(p, X)[0]

def kmax_of(row):
    lo, hi = 0.0, 1.5
    if has_price(params(row, hi)):
        return hi, True
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if has_price(params(row, mid)) else (lo, mid)
    return lo, False

def scan(kind, row, grid):
    vals = []
    for k in grid:
        try: vals.append(gap(kind, row, k))
        except Exception: vals.append(np.nan)
    vals = np.array(vals)
    for a, b, va, vb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(va) and np.isfinite(vb) and va < 0 <= vb:
            try: return brentq(lambda k: gap(kind, row, k), a, b, xtol=1e-3), vals
            except Exception: return 0.5 * (a + b), vals
    return np.nan, vals

def one(args):
    i, row = args
    try:
        if not has_price(params(row, 0.0)):
            return dict(i=i, status="no price at kappa=0")
        km, capped = kmax_of(row)
        grid = np.linspace(0.0, 0.995 * km, 20)
        kF, vF = scan("F", row, grid)
        kU, vU = scan("U", row, grid)
        return dict(i=i, status="root" if np.isfinite(kF) else "no root", kmax=km, capped=capped, kF=kF, kU=kU,
                    GF0=vF[0], GU0=vU[0], nfin=int(np.isfinite(vF).sum()))
    except Exception as e:
        return dict(i=i, status=f"error: {str(e)[:50]}")

if __name__ == "__main__":
    d = pd.read_parquet(Path(__file__).resolve().parents[1] / "uq/output/uq0_draws.parquet")
    sub = d.sample(400, random_state=23)
    with Pool(8) as pool:
        res = pool.map(one, list(sub.iterrows()), chunksize=2)
    o = sub.join(pd.DataFrame(res).set_index("i"))
    o.to_parquet("output/uq1f_draws.parquet")
    print("done", o.status.value_counts().to_dict())
