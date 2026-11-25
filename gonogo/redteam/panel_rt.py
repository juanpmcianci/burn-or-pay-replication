import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from cht import calibrated
from panel import token_types, ratio_draws
thetas=np.linspace(0.15,0.8,8)
for kappa in [0.0,0.1]:
    types=token_types(calibrated(beta=1.1,kappa=kappa,F_max=0.12,F_min=0.0),thetas)
    for sS in [1.0,0.1]:
      for T in [90,1040]:
        d=ratio_draws(types,reps=60,T=T,sigma_S=sS); b,g=d[:,0],d[:,1]
        print(f"k={kappa} sigS={sS} T={T}: beta mean {b.mean():.5f} (sd {b.std():.5f}) gamma med {np.median(g):.4f} P(b>0)={np.mean(b>0):.2f} ratio med {np.median(b/abs(g)):.2f}",flush=True)
