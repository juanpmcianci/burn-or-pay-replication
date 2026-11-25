"""Revision A1: the feasible policy derivative.

A protocol cannot move a uniform yield b at every state; it can only change the burn share w_B, which moves
b(A) = F(A) dw_B. By Lemma 1 this is a state-dependent emission cut, E -> E - F(x) dw_B, so the direct cash flows
still cancel at every state and only the stake effect remains.

Feasible gap at state x0, per unit of current revenue yield (years, same units as the uniform gap G = dlogW/dE):

    G_F(x0) = -(1/F(x0)) d log W(x0) / d w_B      (central difference around w_B0)

Positive means payout beats burn. At CHT's baseline w_B0 = 0 the policy can only move one way (more burn);
the model is smooth in w_B, so the central difference equals the one-sided derivative up to O(dw^2).
We check this against a one-sided difference and an interior base w_B0 = 0.5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gonogo"))
from cht import calibrated, fee, rho_of, solve, theta_of_rho, with_, Params  # noqa: E402,F401

DW = 2e-3      # step in burn share
H = 2e-4       # step in emission yield (uniform benchmark)
CH = ["gamma", "net", "fb"]


def G_uniform(p, x, pin=None, h=H):
    x = np.atleast_1d(x)
    up = solve(with_(p, E=p.E + h), pin=pin).logW(x)
    dn = solve(with_(p, E=p.E - h), pin=pin).logW(x)
    return (up - dn) / (2 * h)


def G_feasible(p, x, pin=None, dw=DW, one_sided=False):
    x = np.atleast_1d(x)
    if one_sided:
        lo = solve(p, pin=pin).logW(x)
        hi = solve(with_(p, w_B=p.w_B + dw), pin=pin).logW(x)
        d = (hi - lo) / dw
    else:
        hi = solve(with_(p, w_B=p.w_B + dw), pin=pin).logW(x)
        lo = solve(with_(p, w_B=p.w_B - dw), pin=pin).logW(x)
        d = (hi - lo) / (2 * dw)
    return -d / fee(x, p)


def theta_path(p):
    return lambda z: theta_of_rho(rho_of(z, p), p)


def dtheta_drho(p, x, h=H):
    r = rho_of(np.atleast_1d(x), p)
    return (theta_of_rho(r + h, p) - theta_of_rho(r - h, p)) / (2 * h)


def M_of(p, x, h=H):
    th0 = theta_path(p)
    pin = {k: th0 for k in CH}
    x = np.atleast_1d(x)
    up = solve(with_(p, D=p.D + h), pin=pin).logW(x)
    dn = solve(with_(p, D=p.D - h), pin=pin).logW(x)
    return (up - dn) / (2 * h)


def channels(p, x, kind="feasible"):
    """Channel gaps: Theta responds in one place of the ODE, the other two pinned at the baseline path."""
    th0 = theta_path(p)
    f = G_feasible if kind == "feasible" else G_uniform
    return {ch: f(p, x, pin={k: th0 for k in CH if k != ch}) for ch in CH}


def x_at_theta(p, target=0.45):
    from scipy.optimize import brentq
    return brentq(lambda z: theta_of_rho(rho_of(np.array([z]), p), p)[0] - target, -6, 6)


def has_price(p):
    try:
        solve(p)
        return True
    except Exception:
        return False
