"""Gate 2, level 0: CHT with two agent types, in the Gordon limit (coefficients frozen at one state).

Types j in {s, h}: mass m_j, staking preference u_j (CHT eq. 27). Holders have u_h = 0 by default.
All quantities are per unit of A*S (productivity times demand shock), so they are stationary.

Given the aggregate staking ratio Theta:
  N = 1 - Theta, r = rho / Theta, k = (r - c)/psi
  theta_j = min(1, k^(1/(psi-1)) N u_j)                     (CHT eq. 27)
  Gamma_j = (r - c) theta_j - theta_j^psi (N u_j)^(1-psi)   (net staking benefit; CHT before eq. 13)
  mu      = mu_A(Theta) - E_net + mu_S                     (Gordon price drift, CHT eqs. 22, A.5)
  w_j     = N [(1-alpha)/(r_f - mu - Gamma_j)]^(1/alpha)    (holdings per agent / (A S); CHT eq. 13)
  Theta   = sum m_j theta_j w_j / sum m_j w_j               (fixed point; CHT eq. 9)
Market cap M = sum m_j w_j (CHT eq. 11).

Welfare of group j (per A0 S0): endowment share e_j of the market cap at the moment of the policy, plus the
present value of the group's participation surplus. At the optimum an agent's flow surplus is
alpha x^(1-alpha) (N A)^alpha (CHT eq. 8 at the optimal x), which grows with A S at mu_A + mu_S:
  Omega_j = e_j M + m_j alpha w_j^(1-alpha) N^alpha / (r_f - mu_A - mu_S).
Endowment shares e_j are baseline holdings, held fixed when the policy changes.

Policy: revenue (fee yield F) split, share w_B burned. rho = E + (1 - w_B) F, E_net = E - w_B F.
As in gate 1, moving yield from payout to burn equals an emission cut, so pay-minus-burn = d/dE.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class P2:
    r_f: float = 0.06
    alpha: float = 0.3
    psi: float = 1.1          # CHT's lock-up cost exponent (their beta)
    c: float = 0.01
    E: float = 0.03
    F: float = 0.03           # fee yield at this state (rho = 6% when all paid)
    w_B: float = 0.0
    mu0: float = 0.02
    kappa: float = 0.0        # productivity feedback d mu_A / d Theta
    theta_bar: float = 0.45
    mu_S: float = 0.0
    m_s: float = 0.5          # mass of stakers; holders have mass 1 - m_s
    u_s: float = 1.0
    u_h: float = 0.0


def with_(p, **kw):
    return replace(p, **kw)


def rho(p):
    return p.E + (1 - p.w_B) * p.F


def E_net(p):
    return p.E - p.w_B * p.F


def mu_A(p, Th):
    return p.mu0 + p.kappa * (Th - p.theta_bar)


def _type(p, Th, u):
    """Staking fraction and net benefit for one type, given aggregate Theta."""
    N = 1 - Th
    r = rho(p) / Th
    if u <= 0 or r <= p.c:
        return 0.0, 0.0
    k = (r - p.c) / p.psi
    th = min(1.0, k ** (1 / (p.psi - 1)) * N * u)
    G = (r - p.c) * th - th ** p.psi * (N * u) ** (1 - p.psi)
    return th, G


def state(p, Th):
    N = 1 - Th
    mu = mu_A(p, Th) - E_net(p) + p.mu_S
    out = {}
    for j, u, m in (("s", p.u_s, p.m_s), ("h", p.u_h, 1 - p.m_s)):
        th, G = _type(p, Th, u)
        den = p.r_f - mu - G
        if den <= 0:
            raise ValueError(f"no price: r_f - mu - Gamma_{j} = {den:.4f}")
        w = N * ((1 - p.alpha) / den) ** (1 / p.alpha)
        out[j] = dict(theta=th, Gamma=G, w=w, m=m)
    X = sum(v["m"] * v["w"] for v in out.values())
    Th_impl = sum(v["m"] * v["theta"] * v["w"] for v in out.values()) / X
    return out, X, Th_impl, mu


def equilibrium(p):
    f = lambda Th: state(p, Th)[2] - Th
    lo, hi = 1e-6, 1 - 1e-6
    grid = np.linspace(lo, hi, 400)
    vals = []
    for g in grid:
        try:
            vals.append(f(g))
        except ValueError:
            vals.append(np.nan)
    vals = np.array(vals)
    roots = []
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            roots.append(brentq(f, grid[i], grid[i + 1], xtol=1e-13))
    if len(roots) != 1:
        raise ValueError(f"{len(roots)} equilibria: {np.round(roots, 4)}")
    Th = roots[0]
    out, X, _, mu = state(p, Th)
    return dict(Theta=Th, types=out, M=X, mu=mu, r=rho(p) / Th)


def welfare(p, eq, e):
    """Group welfare per A0 S0 given endowment shares e = {'s': ., 'h': .}."""
    Th = eq["Theta"]
    N = 1 - Th
    disc = p.r_f - mu_A(p, Th) - p.mu_S
    if disc <= 0:
        raise ValueError("surplus PV diverges")
    W = {}
    for j, v in eq["types"].items():
        surplus = v["m"] * p.alpha * v["w"] ** (1 - p.alpha) * N ** p.alpha / disc
        W[j] = dict(endow=e[j] * eq["M"], surplus=surplus, total=e[j] * eq["M"] + surplus)
    return W


def calibrate(p, theta_t=0.45, staker_wealth=1.0):
    """Closed-form calibration with u_h = 0. Holders do not stake, so Theta = (staker wealth share) * theta_s.
    Given the target Theta and the stakers' wealth share lam, theta_s = Theta/lam; u_s follows from CHT eq. 27,
    and the staker mass m_s from the wealth share. Requires lam >= Theta."""
    lam = staker_wealth
    assert p.u_h == 0 and lam >= theta_t
    th_s = theta_t / lam
    N = 1 - theta_t
    r = rho(p) / theta_t
    k = (r - p.c) / p.psi
    u_s = th_s / (k ** (1 / (p.psi - 1)) * N)
    q = with_(p, u_s=u_s, m_s=0.5)
    out, _, _, _ = state(q, theta_t)
    ws, wh = out["s"]["w"], out["h"]["w"]
    m_s = 1.0 if lam >= 1 else lam * wh / (lam * wh + (1 - lam) * ws)
    q = with_(q, m_s=m_s)
    eq = equilibrium(q)
    assert abs(eq["Theta"] - theta_t) < 1e-8, eq["Theta"]
    return q


def gaps(p, h=1e-5):
    """Pay-minus-burn per unit of revenue yield (d/dE), in years, normalised by baseline market cap."""
    eq0 = equilibrium(p)
    M0 = eq0["M"]
    e = {j: v["m"] * v["w"] / M0 for j, v in eq0["types"].items()}
    up, dn = with_(p, E=p.E + h), with_(p, E=p.E - h)
    equ, eqd = equilibrium(up), equilibrium(dn)
    Wu, Wd = welfare(up, equ, e), welfare(dn, eqd, e)
    W0 = welfare(p, eq0, e)
    d = lambda a, b: (a - b) / (2 * h) / M0
    out = dict(Theta=eq0["Theta"], r=eq0["r"], e_s=e["s"], m_s=p.m_s,
               theta_s=eq0["types"]["s"]["theta"], Gamma_s=eq0["types"]["s"]["Gamma"],
               dTheta_drho=(equ["Theta"] - eqd["Theta"]) / (2 * h),
               G_M=(np.log(equ["M"]) - np.log(eqd["M"])) / (2 * h))
    for j in ("s", "h"):
        out[f"G_{j}"] = d(Wu[j]["total"], Wd[j]["total"])
        out[f"G_{j}_endow"] = d(Wu[j]["endow"], Wd[j]["endow"])
        out[f"G_{j}_surplus"] = d(Wu[j]["surplus"], Wd[j]["surplus"])
        out[f"S0_{j}"] = W0[j]["surplus"] / M0
    out["G_tot"] = out["G_s"] + out["G_h"]
    return out
