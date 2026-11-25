"""RT v2: physical drift of log A is mu0 - sigma^2/2 = -0.041 at baseline. Do the hurdles survive when the median
platform actually grows (sigma_A = 0.15: drift +0.009; mu0 = 0.08: drift +0.019)?"""
import sys, numpy as np, pandas as pd
R=__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"..")
sys.path.insert(0,R); sys.path.insert(0,R+"/../gonogo")
from centred import *
from feasible import x_at_theta
from calibB import closed_form
from multiprocessing import Pool
def job(a):
    psi,rf,kw=a
    p=calibrated(beta=psi,kappa=0.0,r_f=rf,**{**dict(F_max=0.12,F_min=0.0),**kw}); x=x_at_theta(p)
    try:
        kL=hurdle(p,x,"F")[0]; kE=hurdle(p,x,"U")[0]
    except Exception as e: return dict(psi=psi,rf=rf,**kw,err=str(e)[:40])
    return dict(psi=psi,rf=rf,**kw,logA_drift=p.mu0-p.sigma_A**2/2,kE=kE,kL=kL,LE=kL/kE)
if __name__=="__main__":
    jobs=[(psi,rf,kw) for psi,rf in [(1.5,0.2),(11.2,0.2)] for kw in [dict(sigma_A=0.15),dict(mu0=0.08)]]
    with Pool(4) as pool: res=pool.map(job,jobs)
    print(pd.DataFrame(res).round(3).to_string(index=False))
