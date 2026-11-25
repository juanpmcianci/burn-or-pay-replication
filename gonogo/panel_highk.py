"""Panel regression in an economy where payout clearly beats or nearly ties burn (high feedback)."""
import numpy as np
from cht import calibrated
from panel import token_types, ratio_draws
thetas = np.linspace(0.15, 0.8, 8)
for kappa in [0.0, 0.2]:
    base = calibrated(beta=1.1, kappa=kappa, F_max=0.12, F_min=0.0, r_f=0.10)
    try:
        types = token_types(base, thetas)
    except Exception as e:
        print(kappa, e); continue
    for T in [90, 1040]:
        d = ratio_draws(types, reps=200 if T < 1000 else 60, T=T)
        b, g = d[:, 0], d[:, 1]; ratio = b/np.abs(g)
        q = np.percentile(ratio, [5, 50, 95])
        print(f"r_f=0.10 kappa={kappa} T={T}: beta med {np.median(b):.4f} gamma med {np.median(g):.4f} "
              f"P(beta>0)={np.mean(b>0):.2f} ratio 5/50/95 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}", flush=True)
