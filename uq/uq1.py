"""UQ level 1 (UQ_PLAN.md): tie point kappa* from the full lifecycle ODE (gonogo/cht.py) on a random subsample of the
level-0 draws. The lifecycle fee path is logistic in log A with F(x=0) equal to the draw's fee yield (F_min=0,
F_max=2F), so the state x=0 reproduces the level-0 state. kappa* = root of G2 = dlogW/dE at x=0 in [0, kappa_max),
where kappa_max is the largest feedback with a price; 'none' if G2 < 0 up to kappa_max.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gonogo"))
from cht import Params, solve, u_for, with_  # noqa: E402

H = 2e-4
NSUB = 60


def params(row, kappa):
    E = row.Eshare * row.rho
    F = row.rho - E
    return Params(r_f=row.r_f, alpha=row.alpha, beta=row.psi, c=row.c, u=u_for(row.Theta, row.rho, row.psi, row.c),
                  E=E, F_max=2 * F, F_min=0.0, x0=0.0, s=0.75, mu0=row.mu0, kappa=kappa, theta_bar=row.Theta,
                  sigma_A=0.35, mu_S=row.mu_S)


def G2(row, kappa):
    p = params(row, kappa)
    x = np.array([0.0])
    up = solve(with_(p, E=p.E + H)).logW(x)[0]
    dn = solve(with_(p, E=p.E - H)).logW(x)[0]
    return (up - dn) / (2 * H)


def feasible(row, k):
    try:
        solve(params(row, k)); return True
    except Exception:
        return False


def one(args):
    i, row = args
    try:
        if not feasible(row, 0.0):
            return dict(i=i, status="no price at kappa=0")
        lo, hi = 0.0, 1.5
        if feasible(row, hi):
            kmax = hi
        else:
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if feasible(row, mid) else (lo, mid)
            kmax = lo
        last = None
        for f in (0.995, 0.98, 0.95, 0.9):          # step back from the existence boundary if the solver fails there
            try:
                g_hi = G2(row, kmax * f)
            except Exception as e:
                last = e
                continue
            if g_hi < 0:
                return dict(i=i, status="none", kmax=kmax, k1=np.nan, edge=f)
            k1 = brentq(lambda k: G2(row, k), 0.0, kmax * f, xtol=2e-3)
            return dict(i=i, status="root", kmax=kmax, k1=k1, edge=f)
        return dict(i=i, status=f"error near boundary: {str(last)[:50]}", kmax=kmax)
    except Exception as e:
        return dict(i=i, status=f"error: {str(e)[:60]}")


if __name__ == "__main__":
    d = pd.read_parquet(Path(__file__).with_name("output") / "uq0_draws.parquet")
    sub = d.sample(NSUB, random_state=11)
    with Pool(6) as pool:
        res = pool.map(one, list(sub.iterrows()))
    r = pd.DataFrame(res).set_index("i")
    out = sub.join(r)
    out.to_parquet(Path(__file__).with_name("output") / "uq1_draws.parquet")
    print(out.status.value_counts().to_string())
    ok = out[out.status == "root"]
    print(f"\nroots: {len(ok)} of {len(out)}")
    if len(ok):
        from scipy.stats import spearmanr
        print("level-1 kappa* quantiles 5/50/95%:", np.round(ok.k1.quantile([0.05, 0.5, 0.95]).values, 3))
        print("level-0 kappa* on same draws 5/50/95%:", np.round(ok.kstar.quantile([0.05, 0.5, 0.95]).values, 3))
        print("median (level1 - level0):", round(float((ok.k1 - ok.kstar).median()), 3),
              " Spearman:", round(spearmanr(ok.k1, ok.kstar)[0], 3))
    for t in (0.1, 0.2, 0.3):
        l1 = np.where(out.status == "root", out.k1 < t, np.where(out.status == "none", False, np.nan))
        m = ~np.isnan(l1.astype(float))
        agree = np.mean((out.kstar[m] < t).values == l1[m].astype(bool))
        print(f"agreement on 'kappa* < {t}' (draws with a verdict): {agree:.3f}  (n={m.sum()})")
