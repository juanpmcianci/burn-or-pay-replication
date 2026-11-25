"""Global comparison log W(omega=1) - log W(omega=0) at the reference state, and the gap's sign at omega in {0, .5, .9}."""
import numpy as np, pandas as pd
from centred import *
from feasible import x_at_theta
rows = []
for psi, rf in [(1.5, 0.20), (11.2, 0.20)]:
    p0 = calibrated(beta=psi, kappa=0.0, F_max=0.12, F_min=0.0, r_f=rf); x = x_at_theta(p0)
    for k in [0.0, 0.3, 0.6]:
        q = with_(p0, kappa=k)
        lw = {w: solve_c(with_(q, w_B=w), p0)(x)[0] for w in [0.0, 1.0]}
        g = {}
        for w in [0.0, 0.5, 0.9]:
            up = solve_c(with_(q, w_B=w + 2e-3), p0)(x)[0]; dn = solve_c(with_(q, w_B=w - 2e-3), p0)(x)[0]
            g[w] = -(up - dn) / 4e-3 / fee(np.array([x]), q)[0]
        rows.append(dict(psi=psi, rf=rf, kappa=k, burn_minus_pay=lw[1.0] - lw[0.0], G0=g[0.0], G05=g[0.5], G09=g[0.9]))
d = pd.DataFrame(rows); d.to_csv("output/global_omega.csv", index=False); print(d.round(3).to_string(index=False))
