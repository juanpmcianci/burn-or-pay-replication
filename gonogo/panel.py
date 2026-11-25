"""Proposition 5, part (i): CHT Table 4 regression run on panels simulated from the model.

Each token i has its own staking preference u_i (cross-sectional Theta) and starts at
a random lifecycle stage x_i0 = log A_i0. Within a token, Theta moves only because A
moves the fee yield F(A). Dynamics (CHT eq. 1 and the demand shock S):

    dx   = (mu_A(Theta) - sigma_A^2/2) dt + sigma_A dZ^A
    dlnS = (mu_S - sigma_S^2/2) dt + sigma_S (sqrt(m) dZ^M + sqrt(1-m) dZ^i)
    ln P = ln W(x) + ln S - ln Q,   ln Cap = ln W(x) + ln S

Regression (CHT Table 4, col. 6 analogue, weekly, two-way FE):
    r_{i,t+1} = a_i + tau_t + beta Theta_{i,t} + gamma lnCap_{i,t} + e
V1 reads beta/|gamma| as the long-run effect of Theta on log value (V_Theta).
No risk premia: simulated under the model measure.
"""
from __future__ import annotations

import numpy as np

from cht import calibrated, solve, theta_of_rho, rho_of, mu_A, with_, u_for

DT = 1 / 52


def token_types(base, theta_targets):
    """One solved economy per staking preference; theta_targets = Theta at rho = 6%."""
    types = []
    for tt in theta_targets:
        p = with_(base, u=u_for(tt, 0.06, base.beta, base.c), theta_bar=tt)
        s = solve(p)
        grid = np.linspace(-6.5, 6.5, 2601)
        th = theta_of_rho(rho_of(grid, p), p)
        types.append(dict(p=p, grid=grid, logW=s.logW(grid), theta=th, muA=mu_A(th, p),
                          E_net=p.E - p.w_B * 0))
    return types


def simulate(types, N=66, T=90, sigma_S=1.0, m=0.5, x_range=(-3, 3), burn=0, rng=None):
    rng = np.random.default_rng(rng)
    k = rng.integers(0, len(types), N)
    x = rng.uniform(*x_range, N)
    lnS = np.zeros(N)
    TT = T + 1 + burn
    TH = np.empty((TT, N)); LC = np.empty((TT, N)); LP = np.empty((TT, N))
    sA = types[0]["p"].sigma_A
    muS = types[0]["p"].mu_S
    for t in range(TT):
        th = np.empty(N); lw = np.empty(N); ma = np.empty(N); en = np.empty(N)
        for j, ty in enumerate(types):
            idx = k == j
            if not idx.any():
                continue
            xi = np.clip(x[idx], ty["grid"][0], ty["grid"][-1])
            th[idx] = np.interp(xi, ty["grid"], ty["theta"])
            lw[idx] = np.interp(xi, ty["grid"], ty["logW"])
            ma[idx] = np.interp(xi, ty["grid"], ty["muA"])
            en[idx] = ty["p"].E
        TH[t] = th; LC[t] = lw + lnS
        LP[t] = lw + lnS - (en * t * DT)
        zm = rng.standard_normal()
        x = x + (ma - 0.5 * sA ** 2) * DT + sA * np.sqrt(DT) * rng.standard_normal(N)
        lnS = lnS + (muS - 0.5 * sigma_S ** 2) * DT + sigma_S * np.sqrt(DT) * (
            np.sqrt(m) * zm + np.sqrt(1 - m) * rng.standard_normal(N))
    TH, LC, LP = TH[burn:], LC[burn:], LP[burn:]
    R = LP[1:] - LP[:-1]                     # r_{t+1}
    return R, TH[:-1], LC[:-1]


def twoway_fe(Y, *Xs):
    """OLS with token and time fixed effects on a balanced panel (T x N arrays)."""
    def demean(A):
        return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
    y = demean(Y).ravel()
    X = np.column_stack([demean(a).ravel() for a in Xs])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def token_fe(Y, *Xs):
    def demean(A):
        return A - A.mean(0, keepdims=True)
    y = demean(Y).ravel()
    X = np.column_stack([demean(a).ravel() for a in Xs])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def ratio_draws(types, reps=400, fe=twoway_fe, **kw):
    out = []
    rng = np.random.default_rng(12345)
    for r in range(reps):
        R, TH, LC = simulate(types, rng=rng, **kw)
        b, g = fe(R, TH, LC)
        out.append((b, g))
    return np.array(out)


if __name__ == "__main__":
    import sys
    thetas = np.linspace(0.15, 0.8, 8)
    for kappa in [0.0, 0.05, 0.1]:
        base = calibrated(beta=1.1, kappa=kappa, F_max=0.12, F_min=0.0)
        types = token_types(base, thetas)
        for T in [90, 208, 1040]:
            d = ratio_draws(types, reps=300 if T < 1000 else 100, T=T)
            b, g = d[:, 0], d[:, 1]
            ratio = b / np.abs(g)
            q = np.percentile(ratio, [5, 50, 95])
            print(f"kappa={kappa} T={T}: beta med {np.median(b):.4f}  gamma med {np.median(g):.4f} "
                  f"P(beta>0)={np.mean(b>0):.2f} P(gamma<0)={np.mean(g<0):.2f}  beta/|gamma| 5/50/95% = "
                  f"{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}", flush=True)
