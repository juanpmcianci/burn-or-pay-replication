"""Revision A1, second pass (after red-team v1): lifecycle feedback centred on the baseline path.

Draft-1 lifecycle runs used mu_A = mu0 + kappa (Theta - theta_bar) with theta_bar fixed, so raising kappa also
changed baseline growth at every state. Here, as in Proposition 1's centring, feedback acts only on stake the
policy induces:
    mu_A(z) = mu0 + kappa (Theta(z) - Theta_0(z)),
where Theta_0 is the baseline (omega = 0, E given) staking path. The base economy does not depend on kappa, the gap
is affine in kappa, and the hurdle is kappa* = -G(0) / (G(1) - G(0)).
Solver: gonogo/cht.py's ODE with this drift (adapted from the red-team variant revision/redteam_v1/rt.py).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gonogo"))
from cht import calibrated, fee, rho_of, theta_of_rho, gamma, with_, Params  # noqa

def solve_c(p, base, x_lo=-7.0, x_hi=7.0, pin=None, n=801, tol=1e-8, full=False):
    """base: Params of the baseline economy (defines Theta_0). pin: dict channel -> Theta path (as cht.solve)."""
    s2 = p.sigma_A ** 2
    th0 = lambda z: theta_of_rho(rho_of(z, base), base)

    def parts(x):
        th = theta_of_rho(rho_of(x, p), p)
        t_g, t_n, t_f = th, th, th
        if pin:
            t_g = pin["gamma"](x) if "gamma" in pin else th
            t_n = pin["net"](x) if "net" in pin else th
            t_f = pin["fb"](x) if "fb" in pin else th
        mA = p.mu0 + p.kappa * (t_f - th0(x))
        G = gamma(rho_of(x, p), t_g, p)
        En = p.E - p.w_B * fee(x, p) - 0.0
        return mA, G + p.D, En, t_n

    def rhs(x, y):
        w, wx = y
        w = np.maximum(w, 1e-12)
        mA, G, En, tn = parts(x)
        flow = (mA + G - En + p.mu_S - p.r_f) * w + (1 - p.alpha) * (1 - tn) ** p.alpha * w ** (1 - p.alpha)
        return np.vstack([wx, -(2.0 / s2) * ((mA + 0.5 * s2) * wx + flow)])

    def gw(x):
        mA, G, En, tn = parts(x)
        den = p.r_f - (mA - En + p.mu_S) - G
        if np.any(den <= 0):
            raise ValueError(f"no price: min denominator {np.min(den):.4f}")
        return (1 - tn) * ((1 - p.alpha) / den) ** (1 / p.alpha)

    xs = np.linspace(x_lo, x_hi, n)
    w0 = gw(xs)
    sol = solve_bvp(rhs, lambda a, b: np.array([a[0] - gw(np.array([x_lo]))[0], b[0] - gw(np.array([x_hi]))[0]]),
                    xs, np.vstack([w0, np.gradient(w0, xs)]), tol=tol, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    if full:
        return sol
    return lambda z: np.log(sol.sol(np.atleast_1d(z))[0]) + np.atleast_1d(z)

DW, H = 2e-3, 2e-4

def gap(p, x, kind="F", kappa=None, pin=None, **kw):
    """Gap at x with feedback slope kappa around the baseline path of p (p.kappa ignored if kappa given)."""
    base = with_(p, kappa=0.0)
    q = with_(p, kappa=p.kappa if kappa is None else kappa)
    if kind == "F":
        up = solve_c(with_(q, w_B=q.w_B + DW), base, pin=pin, **kw)(x)
        dn = solve_c(with_(q, w_B=q.w_B - DW), base, pin=pin, **kw)(x)
        return (-(up - dn) / (2 * DW) / fee(np.atleast_1d(x), q))[0]
    up = solve_c(with_(q, E=q.E + H), base, pin=pin, **kw)(x)
    dn = solve_c(with_(q, E=q.E - H), base, pin=pin, **kw)(x)
    return ((up - dn) / (2 * H))[0]

def hurdle(p, x, kind="F", k1=0.2, **kw):
    g0 = gap(p, x, kind, kappa=0.0, **kw)
    g1 = gap(p, x, kind, kappa=k1, **kw)
    slope = (g1 - g0) / k1
    return -g0 / slope, g0, slope
