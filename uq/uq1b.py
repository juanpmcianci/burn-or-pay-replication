"""UQ level 1, revision (UQ_PLAN amendment D4): grid scan of the gap G2(kappa) on [0, 0.999*kappa_max] for the same
60 draws as uq1.py; first sign change from - to + refined with brentq. G2 is not monotone in kappa."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
from scipy.optimize import brentq
from uq1 import G2, feasible

def kmax_of(row):
    lo, hi = 0.0, 1.5
    if feasible(row, hi):
        return hi, True
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feasible(row, mid) else (lo, mid)
    return lo, False

def one(args):
    i, row = args
    try:
        if not feasible(row, 0.0):
            return dict(i=i, status="no price at kappa=0")
        km, capped = kmax_of(row)
        grid = np.linspace(0.0, 0.999 * km, 20)
        vals = []
        for k in grid:
            try: vals.append(G2(row, k))
            except Exception: vals.append(np.nan)
        vals = np.array(vals)
        for a, b, va, vb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if np.isfinite(va) and np.isfinite(vb) and va < 0 <= vb:
                return dict(i=i, status="root", kmax=km, capped=capped, k1=brentq(lambda k: G2(row, k), a, b, xtol=1e-3))
        return dict(i=i, status="no root on grid", kmax=km, capped=capped, k1=np.nan, nfinite=int(np.isfinite(vals).sum()))
    except Exception as e:
        return dict(i=i, status=f"error: {str(e)[:50]}")

if __name__ == "__main__":
    out_dir = Path(__file__).with_name("output")
    d = pd.read_parquet(out_dir / "uq0_draws.parquet")
    sub = d.sample(60, random_state=11)
    with Pool(6) as pool:
        res = pool.map(one, list(sub.iterrows()))
    o = sub.join(pd.DataFrame(res).set_index("i"))
    o.to_parquet(out_dir / "uq1b_draws.parquet")
    print(o.status.value_counts().to_string())
    ok = o[o.status == "root"]
    from scipy.stats import spearmanr
    diff = ok.k1 - ok.kstar
    print(f"\nroots {len(ok)} of {len(o)}; kmax capped at 1.5 in {int(o.capped.fillna(False).sum())} draws")
    print("lifecycle kappa* (roots) 5/50/95%:", np.round(ok.k1.quantile([.05, .5, .95]).values, 3),
          "| frozen on same draws:", np.round(ok.kstar.quantile([.05, .5, .95]).values, 3))
    print(f"difference: median {diff.median():+.3f}, mean |diff| {diff.abs().mean():.3f}, 95th pct {diff.quantile(.95):+.3f}, max {diff.max():+.3f}; Spearman {spearmanr(ok.k1, ok.kstar)[0]:.3f}")
    print("lifecycle kappa*/r (roots) 5/50/95%:", np.round((ok.k1 / ok.r).quantile([.05, .5, .95]).values, 2),
          "| min k1 - r:", round(float((ok.k1 - ok.r).min()), 3))
    nr = o[o.status == "no root on grid"]
    print(f"no root on grid: {len(nr)} (kmax below r in {int((nr.kmax < nr.r).sum())} of them)")
