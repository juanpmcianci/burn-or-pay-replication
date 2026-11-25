import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from scipy.optimize import brentq
from cht import calibrated, solve
from sweep import at_mid
def mk(beta,k,rf,**kw): return calibrated(beta=beta,kappa=k,F_max=0.12,F_min=0.0,r_f=rf,**kw)
def feas(beta,rf,k,**kw):
    try: solve(mk(beta,k,rf,**kw)); return True
    except Exception: return False
for beta in [1.1,2.0,11.2]:
  for rf in [0.12,0.15,0.20,0.30]:
    lo,hi=0.0,2.0
    if not feas(beta,rf,0.0): print(beta,rf,"infeasible at k=0"); continue
    for _ in range(18):
        m=0.5*(lo+hi); lo,hi=(m,hi) if feas(beta,rf,m) else (lo,m)
    g=lambda k: at_mid(mk(beta,k,rf))[0]
    M=at_mid(mk(beta,0,rf),field='D')[0]-at_mid(mk(beta,0,rf))[0]
    ghi=g(lo*0.999)
    if ghi>0:
        ks=brentq(g,0,lo*0.999,xtol=1e-4); print(f"psi={beta} rf={rf}: kmax={lo:.3f} k*={ks:.3f} M(k=0)={M:.1f} G2(k=0)={g(0):.2f}")
    else: print(f"psi={beta} rf={rf}: kmax={lo:.3f} no k* (G2 at kmax {ghi:.1f}) M={M:.1f}")
