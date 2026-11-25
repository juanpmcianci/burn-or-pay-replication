"""UQ level 0 (UQ_PLAN.md): tie point kappa* in CHT's frozen state (Gordon limit), over the pre-registered priors.

Policy: pay-minus-burn per unit revenue yield = d log W / dE (gate 1 identity). At the state, feedback is centred at
Theta0, so G(kappa) = G0 + kappa*b exactly and kappa* = -G0/b. Checked against the closed form
kappa* = r + Gamma/(1-Theta) + 1/(M(1-Theta)), M = 1/(alpha D).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261109)
N = 20000
H = 1e-6


def draw(n=N, rng=RNG):
    return pd.DataFrame(dict(
        Theta=rng.uniform(0.27, 0.60, n), rho=rng.uniform(0.03, 0.10, n), Eshare=rng.uniform(0.2, 0.8, n),
        psi=np.exp(rng.uniform(np.log(1.05), np.log(12), n)), alpha=rng.uniform(0.15, 0.60, n),
        r_f=rng.uniform(0.04, 0.15, n), c=rng.uniform(0.0, 0.02, n), mu0=rng.uniform(0.0, 0.04, n),
        mu_S=rng.uniform(-0.02, 0.02, n)))


def u_for(Th, rho, psi, c):
    k = (rho / Th - c) / psi
    return Th / ((1 - Th) * k ** (1 / (psi - 1)))


def theta_of_rho(rho, psi, c, u):
    lo, hi = np.full_like(rho, 1e-12), np.full_like(rho, 1 - 1e-12)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f = mid * (psi * (mid / ((1 - mid) * u)) ** (psi - 1) + c) - rho
        hi = np.where(f > 0, mid, hi); lo = np.where(f > 0, lo, mid)
    return 0.5 * (lo + hi)


def logW(d, h, kappa):
    rho = d.rho + h
    E_net = d.Eshare * d.rho + h
    Th = theta_of_rho(rho.values, d.psi.values, d.c.values, d.u.values)
    G = rho - d.c * Th - Th ** d.psi * ((1 - Th) * d.u) ** (1 - d.psi)
    muA = d.mu0 + kappa * (Th - d.Theta)
    D = d.r_f - muA + E_net - d.mu_S - G
    return np.log(1 - Th) + (np.log(1 - d.alpha) - np.log(D)) / d.alpha, D


def run(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["u"] = u_for(d.Theta, d.rho, d.psi, d.c)
    _, D0 = logW(d, 0.0, 0.0)
    d["D"] = D0
    d = d[d.D > 0].copy()
    g0 = (logW(d, H, 0.0)[0] - logW(d, -H, 0.0)[0]) / (2 * H)
    g1 = (logW(d, H, 1.0)[0] - logW(d, -H, 1.0)[0]) / (2 * H)
    d["G0"] = g0
    d["kstar"] = -g0 / (g1 - g0)
    d["r"] = d.rho / d.Theta
    d["Gamma"] = (d.psi - 1) / d.psi * (d.rho - d.c * d.Theta)
    d["M"] = 1 / (d.alpha * d.D)
    d["kstar_cf"] = d.r + d.Gamma / (1 - d.Theta) + 1 / (d.M * (1 - d.Theta))
    return d


def sobol_binned(d, y="kstar", cols=None, bins=20):
    cols = cols or ["Theta", "rho", "Eshare", "psi", "alpha", "r_f", "c", "mu0", "mu_S"]
    vy = d[y].var()
    out = {}
    for x in cols:
        q = pd.qcut(d[x], bins, labels=False, duplicates="drop")
        out[x] = d.groupby(q)[y].mean().var(ddof=0) / vy   # equal-count bins; population variance of bin means (rev 1: was ddof=1, +5.3%)
    return pd.Series(out)


if __name__ == "__main__":
    raw = draw()
    d = run(raw)
    print(f"draws {N}; with a price {len(d)} ({len(d)/N:.1%}); dropped (no price) {1-len(d)/N:.1%}")
    print("closed-form check: max |kstar - kstar_cf| =", float(np.abs(d.kstar - d.kstar_cf).max()))
    q = d.kstar.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print("kappa* quantiles 5/25/50/75/95%:", np.round(q.values, 3))
    for t in (0.05, 0.1, 0.2, 0.3):
        print(f"  share kappa* < {t}: {np.mean(d.kstar < t):.3f}")
    print("kappa* - r quantiles 5/50/95%:", np.round((d.kstar - d.r).quantile([0.05, 0.5, 0.95]).values, 3))
    print("share with kappa* > r:", round(float(np.mean(d.kstar > d.r)), 4))
    S = sobol_binned(d)
    bs = np.array([sobol_binned(d.sample(len(d), replace=True, random_state=i)).values for i in range(100)])
    lo, hi = np.percentile(bs, [5, 95], axis=0)
    print("\nfirst-order Sobol (binned), 90% bootstrap:")
    for i, (k, v) in enumerate(S.sort_values(ascending=False).items()):
        j = list(S.index).index(k)
        print(f"  {k:7s} {v:.3f}  [{lo[j]:.3f}, {hi[j]:.3f}]")
    print(f"  sum {S.sum():.3f}")
    for a, b in ((0, 0.1), (0, 0.3)):
        k = np.random.default_rng(7).uniform(a, b, len(d))
        print(f"P(payout wins) with kappa ~ U({a},{b}): {np.mean(k > d.kstar.values):.3f}")
    d.to_parquet("output/uq0_draws.parquet")
