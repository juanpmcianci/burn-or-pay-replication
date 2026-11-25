"""Cong-He-Tang (2025) staking economy, extended with a revenue split.

Notation follows CHT. Equation numbers refer to SSRN 4059460 (Feb 2025).

State: x = log A (platform productivity). Unknown: w(x) = W(A)/A, where
W = P Q / S is on-chain wealth scaled by the demand shock (CHT Prop. 3).

Extension (rung 2 of the scoping memo):
  - revenue F(A) (fees, CHT eq. 4) is split: share w_B burned, the rest paid to stakers.
    Paid revenue enters the reward ratio rho; burned revenue lowers net supply growth.
  - D: an outside subsidy paid to stakers (yield on market cap). Used only to
    separate the direct cash-flow effect from the effect of stake.
  - Theta enters the ODE through three channels; each can be pinned at a fixed
    path to decompose policy derivatives:
      'gamma'  : staking net benefit Gamma (lock-up cost and slashing),
      'net'    : active network N = 1 - Theta in transaction convenience (CHT eq. 3),
      'fb'     : productivity feedback mu_A(Theta) (CHT eq. 1).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_bvp


@dataclass(frozen=True)
class Params:
    r_f: float = 0.06       # risk-free rate (CHT r^f)
    alpha: float = 0.3      # transaction-convenience exponent (CHT alpha)
    beta: float = 2.0       # lock-up cost exponent (CHT beta > 1)
    c: float = 0.01         # slashing cost (CHT c_t)
    u: float = 13.27        # staking preference (CHT u); gives Theta = 0.45 at rho = 6%
    E: float = 0.03         # emission rate (CHT E_t)
    F_max: float = 0.06     # fee yield at low productivity
    F_min: float = 0.005    # fee yield at high productivity
    x0: float = 0.0         # log A at fee midpoint
    s: float = 0.75         # width of fee transition in log A
    mu0: float = 0.02       # productivity drift at Theta = theta_bar
    kappa: float = 0.0      # feedback slope d mu_A / d Theta (CHT: weakly positive)
    theta_bar: float = 0.45
    sigma_A: float = 0.35
    mu_S: float = 0.0       # demand-shock drift (CHT mu_S)
    # policy
    w_B: float = 0.0        # share of fee revenue burned (CHT baseline: 0)
    D: float = 0.0          # outside subsidy to stakers, yield on market cap


def fee(x, p: Params):
    """F(A), decreasing in A (CHT: F' <= 0). Logistic in log A."""
    return p.F_min + (p.F_max - p.F_min) / (1.0 + np.exp((x - p.x0) / p.s))


def rho_of(x, p: Params):
    """Reward ratio rho = E + paid fees + subsidy (CHT eq. 4 with split)."""
    return p.E + (1.0 - p.w_B) * fee(x, p) + p.D


def theta_of_rho(rho, p: Params):
    """Solve CHT eq. 16: rho = Theta [beta (Theta/((1-Theta)u))^(beta-1) + c]."""
    rho = np.asarray(rho, dtype=float)
    lo = np.full_like(rho, 1e-9)
    hi = np.full_like(rho, 1 - 1e-9)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = mid * (p.beta * (mid / ((1 - mid) * p.u)) ** (p.beta - 1) + p.c) - rho
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    return 0.5 * (lo + hi)


def lock_cost(theta, p: Params):
    """Aggregate slashing plus lock-up cost per unit wealth: c Theta + Theta^beta ((1-Theta) u)^(1-beta)."""
    return p.c * theta + theta ** p.beta * ((1 - theta) * p.u) ** (1 - p.beta)


def gamma(rho, theta, p: Params):
    """CHT Gamma at the staking optimum: (r - c) Theta - lock-up cost, with r = rho / Theta.
    Equals (beta-1)((r-c)/beta)^(beta/(beta-1)) (1-Theta) u when Theta is the equilibrium."""
    return rho - lock_cost(theta, p)


def mu_A(theta, p: Params):
    return p.mu0 + p.kappa * (theta - p.theta_bar)


def channels(x, p: Params, pin: dict | None = None):
    """Return the Theta used in each channel. `pin` maps channel -> callable x -> Theta."""
    th = theta_of_rho(rho_of(x, p), p)
    out = {"gamma": th, "net": th, "fb": th}
    if pin:
        for k, f in pin.items():
            out[k] = f(x)
    return out


def coefficients(x, p: Params, pin=None):
    th = channels(x, p, pin)
    rho = rho_of(x, p)
    G = gamma(rho, th["gamma"], p)
    E_net = p.E - p.w_B * fee(x, p)
    mA = mu_A(th["fb"], p)
    return mA, G, E_net, th["net"]


def gordon_w(x, p: Params, pin=None):
    """Local Gordon solution (CHT eq. 22) with coefficients frozen at x."""
    mA, G, E_net, thN = coefficients(x, p, pin)
    mu = mA - E_net + p.mu_S
    den = p.r_f - mu - G
    if np.any(den <= 0):
        raise ValueError(f"Gordon denominator non-positive: min {np.min(den):.4f}")
    return (1 - thN) * ((1 - p.alpha) / den) ** (1 / p.alpha)


def solve(p: Params, pin=None, x_lo=-7.0, x_hi=7.0, n=801, tol=1e-8):
    """Solve CHT ODE (23) for w(x) = W/A on [x_lo, x_hi] with Gordon boundary values.

    In x = log A, with W = A w, eq. (23) divided by A reads
      0 = mu_A (w + w') + sigma^2/2 (w' + w'') + [Gamma - E_net + mu_S - r_f] w
          + (1 - alpha) (1 - Theta)^alpha w^(1 - alpha).
    """
    s2 = p.sigma_A ** 2

    def rhs(x, y):
        w, wx = y
        w = np.maximum(w, 1e-12)
        mA, G, E_net, thN = coefficients(x, p, pin)
        flow = (mA + G - E_net + p.mu_S - p.r_f) * w + (1 - p.alpha) * (1 - thN) ** p.alpha * w ** (1 - p.alpha)
        wxx = -(2.0 / s2) * ((mA + 0.5 * s2) * wx + flow)
        return np.vstack([wx, wxx])

    wl = gordon_w(np.array([x_lo]), p, pin)[0]
    wh = gordon_w(np.array([x_hi]), p, pin)[0]

    def bc(ya, yb):
        return np.array([ya[0] - wl, yb[0] - wh])

    xs = np.linspace(x_lo, x_hi, n)
    w0 = gordon_w(xs, p, pin)
    y0 = np.vstack([w0, np.gradient(w0, xs)])
    sol = solve_bvp(rhs, bc, xs, y0, tol=tol, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return Solution(p, sol, pin)


class Solution:
    def __init__(self, p: Params, sol, pin):
        self.p, self.sol, self.pin = p, sol, pin

    def w(self, x):
        return self.sol.sol(x)[0]

    def logW(self, x):
        return np.log(self.w(x)) + x

    def theta(self, x):
        return theta_of_rho(rho_of(x, self.p), self.p)

    def mu(self, x):
        """Expected price appreciation (CHT eq. A.5) in x coordinates."""
        p = self.p
        w, wx = self.sol.sol(x)
        wxx = self.sol.sol(x, 1)[1]
        mA, G, E_net, _ = coefficients(x, p, self.pin)
        return (1 + wx / w) * mA + 0.5 * p.sigma_A ** 2 * (wx + wxx) / w - E_net + p.mu_S


def with_(p: Params, **kw) -> Params:
    return replace(p, **kw)


def u_for(theta_t: float, rho_t: float, beta: float, c: float) -> float:
    """Staking preference u that puts Theta = theta_t at rho = rho_t (inverts CHT eq. 16)."""
    k = (rho_t / theta_t - c) / beta          # = (Theta/((1-Theta)u))^(beta-1)
    return theta_t / ((1 - theta_t) * k ** (1 / (beta - 1)))


def calibrated(beta=1.25, c=0.01, theta_t=0.45, rho_t=0.06, **kw) -> Params:
    """Baseline: Theta = 0.45 at rho = 6% (CHT Table 1 means: Theta 44-46%, rho 6.3%)."""
    return Params(beta=beta, c=c, u=u_for(theta_t, rho_t, beta, c), **kw)
