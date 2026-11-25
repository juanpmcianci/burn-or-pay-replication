"""Item 16 (review round 4): the solver's price condition.
centred.solve_c rejects a parameter set if the Gordon denominator D = r_f - mu_A + E_net - mu_S - Gamma is <= 0
anywhere on the grid. That is sufficient, not necessary. Near A -> 0 (high-fee states) write the ODE in W:
    1/2 s^2 A^2 W'' + mu A W' - R W + (1-alpha)(A n)^alpha W^(1-alpha) = 0,   R = D + mu = r_f - mu_S + E_net - Gamma.
If D > 0 locally the Gordon solution (w = W/A constant) is the asymptote. If D < 0 but R > 0, the linear part
dominates (w -> infinity), W ~ A^g with g the positive root of 1/2 s^2 g^2 + (mu - s^2/2) g - R = 0, and g < 1
exactly when D < 0; W -> 0 as A -> 0, which is CHT's lower boundary condition. If R <= 0 no solution has W -> 0.
So the condition is: D > 0 at the upper end (Gordon limit) and R > 0 at the lower end. solve_c2 imposes the Gordon
value at the left end when D > 0 there and the Robin condition w_z = (g - 1) w otherwise."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_bvp
from centred import theta_of_rho, rho_of, gamma, fee, with_  # noqa

class NoPrice(ValueError):
    pass

def solve_c2(p, base, x_lo=-7.0, x_hi=7.0, n=801, tol=1e-8, full=False, guess=1.0):
    s2 = p.sigma_A ** 2
    th0 = lambda z: theta_of_rho(rho_of(z, base), base)

    def parts(x):
        th = theta_of_rho(rho_of(x, p), p)
        mA = p.mu0 + p.kappa * (th - th0(x))
        G = gamma(rho_of(x, p), th, p) + p.D
        En = p.E - p.w_B * fee(x, p)
        R = p.r_f - p.mu_S + En - G
        return mA, G, En, th, R

    def rhs(x, y):
        w, wx = y
        w = np.maximum(w, 1e-12)
        mA, G, En, tn, R = parts(x)
        flow = (mA - R) * w + (1 - p.alpha) * (1 - tn) ** p.alpha * w ** (1 - p.alpha)
        return np.vstack([wx, -(2.0 / s2) * ((mA + 0.5 * s2) * wx + flow)])

    xs = np.linspace(x_lo, x_hi, n)
    mA, G, En, th, R = parts(xs)
    den = R - mA
    if den[-1] <= 0:
        raise NoPrice(f"no price: upper-end D {den[-1]:.4f}")
    if R[0] <= 0 and den[0] <= 0:      # with mu_A < 0, D > 0 suffices even if R <= 0
        raise NoPrice(f"no price: lower-end R {R[0]:.4f}")
    gord = lambda i: (1 - th[i]) * ((1 - p.alpha) / den[i]) ** (1 / p.alpha)
    left_gordon = den[0] > 0
    if not left_gordon:
        b = mA[0] - s2 / 2
        g = (-b + np.sqrt(b * b + 2 * s2 * R[0])) / s2     # in (0, 1) since D < 0 < R
    # initial guess: Gordon where den > 0, power-law continuation to the left of the last den <= 0 node
    w0 = np.empty(n)
    ok = den > 0
    w0[ok] = (1 - th[ok]) * ((1 - p.alpha) / den[ok]) ** (1 / p.alpha)
    if not ok.all():
        j = np.where(~ok)[0].max() + 1                      # first node of the right-hand den > 0 block
        if not ok[j:].all():
            # interior den <= 0 pockets to the right: fill by interpolation in log w
            idx = np.arange(n); good = ok & (idx >= j)
            w0[j:] = np.exp(np.interp(idx[j:], idx[good], np.log(w0[good])))
        gg = g if not left_gordon else 0.5
        w0[:j] = w0[j] * np.exp((gg - 1) * (xs[:j] - xs[j]))
    def bc(a, bb):
        left = a[0] - gord(0) if left_gordon else a[1] - (g - 1) * a[0]
        return np.array([left, bb[0] - gord(n - 1)])
    w0 = w0 * guess * np.exp(0.0 * xs)
    sol = solve_bvp(rhs, bc, xs, np.vstack([w0, np.gradient(w0, xs)]), tol=tol, max_nodes=1000000)
    if not sol.success:
        raise RuntimeError(sol.message)
    if np.any(sol.sol(xs)[0] <= 0):
        raise RuntimeError("non-positive w")
    if full:
        return sol
    return lambda z: np.log(sol.sol(np.atleast_1d(z))[0]) + np.atleast_1d(z)

DW, H = 2e-3, 2e-4

def gap2(p, x, kind="F", kappa=0.0, **kw):
    base = with_(p, kappa=0.0); q = with_(p, kappa=kappa)
    if kind == "F":
        up = solve_c2(with_(q, w_B=q.w_B + DW), base, **kw)(x); dn = solve_c2(with_(q, w_B=q.w_B - DW), base, **kw)(x)
        return (-(up - dn) / (2 * DW) / fee(np.atleast_1d(x), q))[0]
    up = solve_c2(with_(q, E=q.E + H), base, **kw)(x); dn = solve_c2(with_(q, E=q.E - H), base, **kw)(x)
    return ((up - dn) / (2 * H))[0]
