import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, cht
from cht import calibrated
from sweep import at_mid
K=float(sys.argv[1])
cht.mu_A = lambda th,p: p.mu0 + K*np.tanh(p.kappa*(th-p.theta_bar)/K)
for beta,rf in [(1.1,0.06),(1.1,0.10),(2.0,0.10)]:
  for k in [0.3,0.6,1.0,2.0]:
    p=calibrated(beta=beta,kappa=k,F_max=0.12,F_min=0.0,r_f=rf)
    try:
        g2,d,x=at_mid(p); print(f"bounded K={K} psi={beta} rf={rf} local k={k}: G2={g2:.2f} V={g2/d:.2f}")
    except Exception as e: print(f"psi={beta} rf={rf} k={k}: --",str(e)[:60])
