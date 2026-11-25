"""Red-team checks. Variant solver: options to
 - pin the lower/upper boundary values (bc_w = (wl, wh)),
 - pin Theta in the drift (state-motion) term mu_A * w_z only ('drift_pin'),
 - give the base path a different feedback from the perturbation ('kb': base kappa; perturbation slope p.kappa)."""
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"..","..","gonogo"))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),".."))
import numpy as np
from scipy.integrate import solve_bvp
from cht import calibrated, fee, rho_of, theta_of_rho, gamma, with_, gordon_w, coefficients
from feasible import x_at_theta

BASE_CAL = dict(F_max=0.12, F_min=0.0)


def make(psi, rf, k, **kw):
    d = dict(BASE_CAL); d.update(kw)
    return calibrated(beta=psi, kappa=k, r_f=rf, **d)


def solve2(p, x_lo=-7.0, x_hi=7.0, bc_w=None, drift_pin=None, kb=None, th_base=None, n=801, tol=1e-8):
    s2 = p.sigma_A ** 2

    def mus(x):
        th = theta_of_rho(rho_of(x, p), p)
        if kb is None:
            mg = p.mu0 + p.kappa * (th - p.theta_bar)
        else:  # base path feedback kb, perturbation slope kappa around th_base
            tb = th_base(x)
            mg = p.mu0 + kb * (tb - p.theta_bar) + p.kappa * (th - tb)
        md = mg if drift_pin is None else p.mu0 + p.kappa * (drift_pin(x) - p.theta_bar)
        return th, mg, md

    def rhs(x, y):
        w, wx = y
        w = np.maximum(w, 1e-12)
        th, mg, md = mus(x)
        G = gamma(rho_of(x, p), th, p)
        En = p.E - p.w_B * fee(x, p)
        flow = (mg + G - En + p.mu_S - p.r_f) * w + (1 - p.alpha) * (1 - th) ** p.alpha * w ** (1 - p.alpha)
        wxx = -(2.0 / s2) * ((md + 0.5 * s2) * wx + flow)
        return np.vstack([wx, wxx])

    def gw(x):
        th, mg, md = mus(x)
        G = gamma(rho_of(x, p), th, p)
        En = p.E - p.w_B * fee(x, p)
        den = p.r_f - (mg - En + p.mu_S) - G
        if np.any(den <= 0):
            raise ValueError("no price")
        return (1 - th) * ((1 - p.alpha) / den) ** (1 / p.alpha)

    if bc_w is None:
        wl, wh = gw(np.array([x_lo]))[0], gw(np.array([x_hi]))[0]
    else:
        wl, wh = bc_w
    xs = np.linspace(x_lo, x_hi, n)
    w0 = gw(xs)
    y0 = np.vstack([w0, np.gradient(w0, xs)])
    sol = solve_bvp(rhs, lambda a, b: np.array([a[0] - wl, b[0] - wh]), xs, y0, tol=tol, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol, (wl, wh)


def logW(sol, x):
    return np.log(sol.sol(np.atleast_1d(x))[0]) + x


def gap(p, x, kind="F", h=None, pin_bc=False, one_sided=False, **kw):
    """kind F: -(1/F) dlogW/dw_B ; kind U: dlogW/dE."""
    base, bcw = solve2(p, **kw)
    bc = bcw if pin_bc else None
    if kind == "F":
        h = h or 2e-3
        up = logW(solve2(with_(p, w_B=p.w_B + h), bc_w=bc, **kw)[0], x)
        if one_sided:
            return (-(up - logW(base, x)) / h / fee(x, p))[0]
        dn = logW(solve2(with_(p, w_B=p.w_B - h), bc_w=bc, **kw)[0], x)
        return (-(up - dn) / (2 * h) / fee(x, p))[0]
    h = h or 2e-4
    up = logW(solve2(with_(p, E=p.E + h), bc_w=bc, **kw)[0], x)
    dn = logW(solve2(with_(p, E=p.E - h), bc_w=bc, **kw)[0], x)
    return ((up - dn) / (2 * h))[0]
