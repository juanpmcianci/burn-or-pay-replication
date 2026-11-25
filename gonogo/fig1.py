"""Check against CHT Figure 1: Theta and mu positively related, with (kappa>0) and without (kappa=0) feedback."""
import numpy as np
from cht import calibrated, solve
for beta in [1.05, 1.1, 1.25, 2.0]:
    for k in [0.0, 0.05, 0.1]:
        p = calibrated(beta=beta, kappa=k, F_max=0.12, F_min=0.0, r_f=0.06 if beta < 2 else 0.10)
        try: s = solve(p)
        except Exception as e: print(f"beta={beta} kappa={k}: {e}"); continue
        xs = np.linspace(-3, 3, 13); th = s.theta(xs); mu = s.mu(xs)
        print(f"beta={beta} kappa={k}: Theta {th.min():.2f}-{th.max():.2f}, mu {mu.min()*100:+.2f}..{mu.max()*100:+.2f} %/yr, corr(Theta,mu) = {np.corrcoef(th, mu)[0,1]:+.2f}")
