"""Fact-check v4 follow-ups: (1) Table 5 price thresholds under the paper's condition (centred2); (2) Prop 2 formula
check for the psi = 2.34 baseline rows; (3) risk-premium case under centred2."""
import numpy as np, sys
from centred import calibrated, with_, fee
from centred2 import gap2
from feasible import x_at_theta
import prop2
def hur(p, x, kind):
    g0 = gap2(p, x, kind, 0.0); g2 = gap2(p, x, kind, 0.2); return -g0 / ((g2 - g0) / 0.2)
for psi, rf, mu0 in [(2.0, 0.06, .02), (11.2, 0.08, .02), (11.2, 0.10, .02), (11.2, 0.12, .02), (11.2, 0.06, -0.12), (1.5, 0.06, -0.12), (2.34, 0.20, .02)]:
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0, s=0.75, mu0=mu0); x = x_at_theta(p)
    try: print(psi, rf, mu0, "kL %.3f kE %.3f" % (hur(p, x, "F"), hur(p, x, "U")))
    except Exception as e: print(psi, rf, mu0, "->", str(e)[:60])
for rf in (0.10, 0.20, 0.30):
    r, w = prop2.job(("base", 2.34, rf, {}))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ["psi","rf","kE_fd","kE_formula","kL_fd","kL_formula","qmin","intm","kstar_monotone_99","status"]})
