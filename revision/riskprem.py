"""Item 19: a risk premium as a drift adjustment. Under a market price of productivity risk lambda, the risk-neutral
drift is mu0 - lambda sigma_A. Compare hurdles at (r_f = 0.20, mu0 = 0.02) with (r_f = 0.06, mu0 = 0.02 - 0.14)."""
from centred import calibrated, hurdle
from feasible import x_at_theta
for psi in (1.5, 11.2):
    for lab, rf, mu0 in [("rf=.20", 0.20, 0.02), ("rf=.06, drift-0.14", 0.06, 0.02 - 0.14)]:
        p = calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0, s=0.75, mu0=mu0)
        x = x_at_theta(p)
        kL = hurdle(p, x, "F")[0]; kE = hurdle(p, x, "U")[0]
        print(psi, lab, f"kL {kL:.3f} kE {kE:.3f}")
