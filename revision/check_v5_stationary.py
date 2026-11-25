"""Check v5, task 5: stationary hurdle under constant-centre feedback mu_A = mu0 + kappa (Theta - theta_bar).
Closed-form Gordon price W/A = n(Theta) ((1-alpha)/D)^(1/alpha), n = 1-Theta, D = r_f - mu_A + E_net - mu_S - Gamma,
rho = E + (1-omega) F, E_net = E - omega F, Gamma = rho - c Theta - L(Theta), L = Theta^beta ((1-Theta)u)^(1-beta).
Numerical hurdle: root in kappa of d log W/dE (central difference, h=1e-6) and of -(1/F) d log W/d omega (h=1e-6).
Claimed: kappa* = (c + L' - alpha Dbar n'/n) / (1 - alpha (Theta - theta_bar) n'/n), Dbar = D at kappa = 0."""
import numpy as np
from scipy.optimize import brentq
from centred import calibrated, theta_of_rho, gamma, with_

def logW(p, F, kappa, tb, E=None, om=0.0):
    E = p.E if E is None else E
    rho = np.array([E + (1 - om) * F]); th = theta_of_rho(rho, p)[0]
    G = gamma(rho, np.array([th]), p)[0]; D = p.r_f - (p.mu0 + kappa * (th - tb)) + (E - om * F) - p.mu_S - G
    if D <= 0: raise ValueError("D<=0")
    return np.log(1 - th) + np.log((1 - p.alpha) / D) / p.alpha, th, D

def gapE(p, F, k, tb, h=1e-6): return (logW(p, F, k, tb, E=p.E + h)[0] - logW(p, F, k, tb, E=p.E - h)[0]) / (2 * h)
def gapW(p, F, k, tb, h=1e-6): return -(logW(p, F, k, tb, om=h)[0] - logW(p, F, k, tb, om=-h)[0]) / (2 * h) / F

def claimed(p, F, tb):
    _, th, Dbar = logW(p, F, 0.0, tb); e = 1e-7
    L = lambda t: t ** p.beta * ((1 - t) * p.u) ** (1 - p.beta)
    Lp = (L(th + e) - L(th - e)) / (2 * e); npn = -1 / (1 - th)
    return (p.c + Lp - p.alpha * Dbar * npn) / (1 - p.alpha * (th - tb) * npn), th, Dbar

lines = []
for psi, rf, E, F, tb in [(1.5, 0.20, 0.03, 0.03, 0.35), (2.34, 0.10, 0.03, 0.05, 0.45), (11.2, 0.15, 0.03, 0.015, 0.50),
                          (1.5, 0.20, 0.03, 0.03, 0.45), (2.0, 0.30, 0.04, 0.06, 0.30)]:
    p = calibrated(beta=psi, r_f=rf, E=E)
    kc, th, Dbar = claimed(p, F, tb)
    lo, hi = -2.0, 5.0
    if th > tb: hi = min(hi, (Dbar / (th - tb)) * (1 - 1e-3))
    if th < tb: lo = max(lo, (Dbar / (th - tb)) * (1 - 1e-3))
    kE = brentq(lambda k: gapE(p, F, k, tb), lo, hi, xtol=1e-12)
    kW = brentq(lambda k: gapW(p, F, k, tb), lo, hi, xtol=1e-12)
    # centred-form (Prop. 1) closed form for reference: c + L' - alpha D n'/n with D at kappa = 0
    lines.append(f"psi={psi} rf={rf} E={E} F={F} theta_bar={tb}: Theta={th:.5f} Dbar={Dbar:.5f} | claimed {kc:.8f} | "
                 f"root dlogW/dE {kE:.8f} | root burn-share gap {kW:.8f} | diff {kE - kc:.2e}, {kW - kc:.2e}")
s = "\n".join(lines); print(s); open("output/check_v5_stationary.txt", "w").write(__doc__ + "\n" + s + "\n")
