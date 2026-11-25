import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from scipy.optimize import brentq
from cht import calibrated, solve
import sweep
from policy import dlogW, dtheta_drho
# r at calibration point: rho_t at Theta=.45. Low-reward calibration: rho_t=0.03 (r=6.7%), E=0.015, fees 0..0.06
def mk(k,rf,beta=1.1,rho_t=0.03,E=0.015,Fmax=0.06,s=0.75):
    return calibrated(beta=beta,kappa=k,rho_t=rho_t,E=E,F_max=Fmax,F_min=0.0,r_f=rf,s=s)
for rf in [0.06,0.10]:
  for k in [0.0,0.05,0.1,0.15,0.2]:
    p=mk(k,rf)
    try:
        g2,d,x=sweep.at_mid(p); print(f"r~6.7% rf={rf} k={k}: G2={g2:.2f} V={g2/d:.2f}")
    except Exception as e: print(f"rf={rf} k={k}: --",str(e)[:50])
# fee schedule variants at baseline psi=1.1 rf=.06 k=0
for Fmax,s in [(0.06,0.75),(0.12,0.3),(0.12,2.0),(0.20,0.75)]:
    p=calibrated(beta=1.1,kappa=0.0,F_max=Fmax,F_min=0.0,s=s,r_f=0.06)
    try:
        g2,d,x=sweep.at_mid(p); print(f"fees Fmax={Fmax} s={s}: G2={g2:.2f} V={g2/d:.2f}")
    except Exception as e: print(Fmax,s,'--',str(e)[:50])
