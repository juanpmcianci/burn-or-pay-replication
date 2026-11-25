"""Checks: (1) the referee's numbers reproduce; (2) central vs one-sided vs interior base w_B0 = 0.5;
(3) frozen-state equality: with a constant fee schedule, feasible = uniform; (4) step-size stability."""
import numpy as np
from feasible import *

x0 = 0.75 * np.log(3)
for rf, k in [(0.06, 0.0), (0.06, 0.1), (0.10, 0.0), (0.10, 0.1), (0.10, 0.2)]:
    p = calibrated(beta=1.1, F_max=0.12, F_min=0.0, r_f=rf, kappa=k)
    gu = G_uniform(p, x0)[0]
    gc = G_feasible(p, x0)[0]
    go = G_feasible(p, x0, one_sided=True)[0]
    gs = G_feasible(p, x0, dw=5e-4)[0]
    q = with_(p, w_B=0.5)
    try:
        xq = x_at_theta(q)   # interior base: evaluate at the state with Theta = 0.45 under w_B = 0.5
        gi = G_feasible(q, xq)[0]; gui = G_uniform(q, xq)[0]
    except Exception as e:
        gi = gui = np.nan
    print(f"rf={rf} k={k}: uniform {gu:7.2f} | feasible central {gc:7.2f} one-sided {go:7.2f} dw=5e-4 {gs:7.2f}"
          f" | base wB=0.5 (Theta=.45 state): uniform {gui:7.2f} feasible {gi:7.2f}")

# frozen-state check: constant fees -> uniform and feasible coincide
p = calibrated(beta=1.1, F_max=0.03, F_min=0.03, r_f=0.10, kappa=0.1)
print("constant fee schedule:", G_uniform(p, 0.0)[0], G_feasible(p, 0.0)[0])
