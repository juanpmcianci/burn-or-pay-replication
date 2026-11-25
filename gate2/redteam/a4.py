import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np, itertools
from hetero import *
print("== untested params: alpha, c, mu0, E (kappa 0/0.1) ==")
n=0;conf=0;mpos=0;worst=-1e9
for a,c,mu0,E,psi,lam,k,rf in itertools.product([0.15,0.3,0.5,0.7],[0.0,0.01,0.03],[0.0,0.02,0.04],[0.0,0.03,0.06],[1.1,2.0],[0.5,0.9],[0,0.1],[0.06,0.10]):
    try:
        p=calibrate(P2(alpha=a,c=c,mu0=mu0,E=E,psi=psi,kappa=k,r_f=rf),staker_wealth=lam); g=gaps(p)
    except Exception: continue
    n+=1; conf+= g['G_s']>0>g['G_M']; mpos+=g['G_M']>0; worst=max(worst,g['G_s'])
    if g['G_s']>0 or g['G_M']>0: print(dict(a=a,c=c,mu0=mu0,E=E,psi=psi,lam=lam,k=k,rf=rf),{x:round(g[x],2) for x in ['G_M','G_s','G_h']})
print(n,"feasible; conflict",conf,"G_M>0",mpos,"max G_s",round(worst,2))
print("== holders stake more: u_h/u_s -> 1 ==")
for psi,rf in [(1.1,0.06),(2.0,0.10)]:
  for lam in [0.5,0.75]:
    base=calibrate(P2(psi=psi,r_f=rf),staker_wealth=lam)
    for fr in [0.5,0.7,0.8,0.9,0.95,0.99]:
        try:
            g=gaps(with_(base,u_h=fr*base.u_s)); print(psi,rf,lam,fr,{x:round(g[x],2) for x in ['Theta','G_M','G_s','G_s_endow','G_s_surplus','G_h','e_s']})
        except Exception as e: print(psi,lam,fr,str(e)[:50])
