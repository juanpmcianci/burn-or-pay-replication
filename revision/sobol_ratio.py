"""Revision A2: Sobol indices for kappa*/r (frozen state), first order (binned, as uq0) and total order (Jansen
estimator on the closed form, Saltelli design). Draws with no price (D <= 0) are dropped: the indices are for the
priced region. Also: kappa*/r quantiles conditional on M <= 100 years, and in the psi subsets (A4)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "uq"))
from uq0 import sobol_binned  # noqa

COLS = ["Theta", "rho", "Eshare", "psi", "alpha", "r_f", "c", "mu0", "mu_S"]
LO = dict(Theta=0.27, rho=0.03, Eshare=0.2, psi=np.log(1.05), alpha=0.15, r_f=0.04, c=0.0, mu0=0.0, mu_S=-0.02)
HI = dict(Theta=0.60, rho=0.10, Eshare=0.8, psi=np.log(12), alpha=0.60, r_f=0.15, c=0.02, mu0=0.04, mu_S=0.02)

def ratio(X):
    Th, rho, Es, lpsi, a, rf, c, mu0, muS = X.T
    psi = np.exp(lpsi)
    G = (psi - 1) / psi * (rho - c * Th)
    D = rf - mu0 + Es * rho - muS - G
    r = rho / Th
    k = r + G / (1 - Th) + a * D / (1 - Th)
    return np.where(D > 0, k / r, np.nan)

def unif(n, rng):
    return np.column_stack([rng.uniform(LO[c], HI[c], n) for c in COLS])

if __name__ == "__main__":
    d = pd.read_parquet(Path(__file__).resolve().parents[1] / "uq/output/uq0_draws.parquet")
    d["ratio"] = d.kstar / d.r
    S1 = sobol_binned(d, y="ratio")
    bs = np.array([sobol_binned(d.sample(len(d), replace=True, random_state=i), y="ratio").values for i in range(100)])
    lo, hi = np.percentile(bs, [5, 95], axis=0)
    # total order, Jansen, on priced region: evaluate on A, B, AB_i; keep rows priced in all
    rng = np.random.default_rng(5)
    n = 200000
    A, B = unif(n, rng), unif(n, rng)
    fA, fB = ratio(A), ratio(B)
    ST = {}
    for j, c in enumerate(COLS):
        ABj = A.copy(); ABj[:, j] = B[:, j]
        fAB = ratio(ABj)
        m = np.isfinite(fA) & np.isfinite(fAB) & np.isfinite(fB)
        V = np.var(np.concatenate([fA[m], fB[m]]))
        ST[c] = 0.5 * np.mean((fA[m] - fAB[m]) ** 2) / V
    print("kappa*/r, frozen state, n =", len(d))
    print(f"{'param':8s} {'S1':>6s} {'90% boot':>15s} {'ST':>6s}")
    for j, c in sorted(enumerate(COLS), key=lambda t: -S1.iloc[t[0]]):
        print(f"{c:8s} {S1[c]:6.3f} [{lo[j]:.3f}, {hi[j]:.3f}] {ST[c]:6.3f}")
    print(f"sum S1 {S1.sum():.3f}; sum ST {sum(ST.values()):.3f}")
    q = lambda s: np.round(s.quantile([.05, .5, .95]).values, 2)
    print("all:", q(d.ratio), "| M <= 100:", q(d.ratio[d.M <= 100]), f"(share kept {np.mean(d.M <= 100):.3f})",
          "| M <= 50:", q(d.ratio[d.M <= 50]), f"(share {np.mean(d.M <= 50):.3f})")
    print("M quantiles 50/95/99:", np.round(d.M.quantile([.5, .95, .99]).values, 1))
    for nm, m in [("psi<1.5", d.psi < 1.5), ("psi<2", d.psi < 2), ("psi>=2", d.psi >= 2), ("psi 8-12", d.psi >= 8)]:
        print(f"{nm}: n={m.sum()} kstar median {d.kstar[m].median():.3f}; kstar/r 5/50/95 {q(d.ratio[m])}")
    for a in [(0.15, 0.3), (0.3, 0.6)]:
        m = d.alpha.between(*a); print(f"alpha in {a}: ratio {q(d.ratio[m])}")
    for a in [(0.04, 0.08), (0.08, 0.15)]:
        m = d.r_f.between(*a); print(f"r_f in {a}: ratio {q(d.ratio[m])}")
