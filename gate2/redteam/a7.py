import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np, itertools
from hetero import *
from scipy.optimize import brentq
def ks(lam=0.75,**kw):
    def g(k,key): return gaps(calibrate(P2(kappa=k,**kw),staker_wealth=lam))[key]
    out={}
    for key in ['G_M','G_tot','G_s','G_h']:
        try: out[key]=brentq(lambda k:g(k,key),0.0,0.5,xtol=1e-4)
        except Exception: out[key]=np.nan
    return out
print("tie points kappa* (lam=.75) for M, total welfare, stakers, holders")
for a,psi,rf,F,m0 in [(0.3,1.1,.06,.03,.02),(0.3,1.1,.10,.03,.02),(0.3,2.0,.10,.03,.02),(0.3,1.1,.06,.015,.02),(0.3,1.1,.06,.045,.02),(0.15,1.1,.06,.03,.02),(0.45,1.1,.06,.03,.02),(0.3,1.1,.06,.03,0.0)]:
    o=ks(alpha=a,psi=psi,r_f=rf,F=F,mu0=m0); print(dict(alpha=a,psi=psi,rf=rf,F=F,mu0=m0),{k:round(v,3) for k,v in o.items()})
print("\nPLAN box at kappa=0.16 baseline alpha=.3 r_f=.06:")
for k in [0.15,0.16,0.17]:
  n=c1=c2=0
  for lam,psi,F in itertools.product([0.45,0.5,0.6,0.75,0.9,1.0],[1.05,1.1,1.25,1.5,1.65],[0.015,0.02,0.03,0.04,0.045]):
    try: g=gaps(calibrate(P2(kappa=k,psi=psi,F=F),staker_wealth=lam))
    except Exception: continue
    n+=1; c1+=(g['G_s']>0>g['G_M']); c2+=(np.sign(g['G_tot'])!=np.sign(g['G_M']))
  print(k,n,"feasible; staker conflict",c1,"; tot vs M disagree",c2)
