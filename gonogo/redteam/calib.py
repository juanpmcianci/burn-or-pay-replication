import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from scipy.optimize import brentq
from cht import calibrated, solve, theta_of_rho, rho_of, gamma
from sweep import at_mid, channels_mid, x_at_theta
# psi giving dTheta/drho = 0.413 at Theta=.45, rho=.06
def dth(beta):
    p=calibrated(beta=beta); h=1e-5
    return (theta_of_rho(np.array([0.06+h]),p)[0]-theta_of_rho(np.array([0.06-h]),p)[0])/(2*h)
for b in [1.1,2,4,6,8,10,12]: print(b, round(dth(b),3))
bstar=brentq(lambda b: dth(b)-0.413, 2, 30); print("psi matching 0.413:", bstar)
for rf in [0.06,0.08,0.10,0.15]:
  for k in [0.0,0.1,0.2,0.3,0.4]:
    p=calibrated(beta=bstar,kappa=k,F_max=0.12,F_min=0.0,r_f=rf)
    try:
        g2,d,x=at_mid(p); ch=channels_mid(p)
        Mdir=ch['direct']
        th=0.45; r=rho_of(np.array([x]),p)[0]/th; G=gamma(rho_of(np.array([x]),p),np.array([th]),p)[0]
        cf=Mdir*(k-r-G/(1-th))-1/(1-th)
        print(f"rf={rf} k={k}: dTh/drho={d:.3f} G2={g2:.3f} V={g2/d:.2f} (gamma {ch['gamma']/d:.2f} net {ch['net']/d:.2f} fb {ch['fb']/d:.2f}) M={Mdir:.1f} Gam/N={G/(1-th):.4f} closed={cf:.2f}")
    except Exception as e: print(f"rf={rf} k={k}: --",str(e)[:50])
