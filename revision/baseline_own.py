"""Items 14, 18: (a) the lock-up exponent psi that gives the staking response estimated on our panel (Theta' ~ 2.3)
at the reference state, and its lifecycle hurdles; (b) the steady-state hurdle under priors on (Theta, r) set from
the panel's interquartile ranges (Theta 0.34-0.56, r 2.4-6.1%)."""
import numpy as np
from scipy.optimize import brentq
from centred import calibrated, hurdle
from feasible import x_at_theta
Th, rho, c = 0.45, 0.06, 0.01; r = rho / Th
thp = lambda psi: 1 / (c + psi * (r - c + (psi - 1) / psi * (rho - c * Th) / (1 - Th)))
for target in (2.2, 2.5):
    print("Theta'", target, "-> psi", round(brentq(lambda s: thp(s) - target, 1.01, 20), 3))
psi = round(brentq(lambda s: thp(s) - 2.3, 1.01, 20), 2); print("baseline psi", psi, "Theta'", round(thp(psi), 3))
for rf in (0.10, 0.20, 0.30):
    try:
        p = calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0, s=0.75); x = x_at_theta(p)
        kL = hurdle(p, x, "F"); kE = hurdle(p, x, "U")
        G = (psi - 1) / psi * (rho - c * Th); D = rf - 0.02 + 0.03 - G; ks = r + G / (1 - Th) + 0.3 * D / (1 - Th)
        print(f"rf {rf}: G0E {kE[1]:.1f} G0 {kL[1]:.1f} sE {kE[2]:.0f} sL {kL[2]:.0f} kstar {ks:.3f} kE {kE[0]:.3f} kL {kL[0]:.3f} ratio {kL[0]/ks:.2f}")
    except Exception as e:
        print(rf, "fail", e)
g = np.random.default_rng(5); N = 200000
Th = g.uniform(0.34, 0.56, N); rr = g.uniform(0.024, 0.061, N); rho = rr * Th
psi_ = np.exp(g.uniform(np.log(1.05), np.log(12), N)); al = g.uniform(0.15, 0.6, N); rf = g.uniform(0.04, 0.15, N)
cc = g.uniform(0, 0.02, N); mu0 = g.uniform(0, 0.04, N); muS = g.uniform(-0.02, 0.02, N); Es = g.uniform(0.2, 0.8, N)
ok = cc < rr
G = (psi_ - 1) / psi_ * (rho - cc * Th); D = rf - mu0 + Es * rho - muS - G; ok &= D > 0
ks = rr + G / (1 - Th) + al * D / (1 - Th); x = (ks / rr)[ok]
print("data priors: n ok", ok.sum(), "kstar/r q05/50/95", np.round(np.quantile(x, [.05, .5, .95]), 2), "| kstar q05/50/95", np.round(np.quantile(ks[ok], [.05, .5, .95]), 3))
