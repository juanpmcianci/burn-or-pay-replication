import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from cht import calibrated, solve
for k in [0.0,0.1,0.3]:
    s=solve(calibrated(beta=11.2,kappa=k,F_max=0.12,F_min=0.0,r_f=0.15)); xs=np.linspace(-3,3,13); th=s.theta(xs); mu=s.mu(xs)
    print(f"psi=11.2 k={k}: Theta {th.min():.2f}-{th.max():.2f} mu {mu.min()*100:+.2f}..{mu.max()*100:+.2f} corr {np.corrcoef(th,mu)[0,1]:+.2f}")
