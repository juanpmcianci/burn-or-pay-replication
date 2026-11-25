import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np
from hetero import *
def run(label,**kw):
    for k in [0,0.05,0.1]:
      row=[]
      for lam in [0.5,0.75,1.0]:
        try:
            p=calibrate(P2(kappa=k,**kw),staker_wealth=lam); g=gaps(p)
            row.append(f"lam{lam}: M{g['G_M']:6.1f} s{g['G_s']:6.1f}(sur{g['G_s_surplus']:5.1f}) h{g['G_h']:6.1f} tot{g['G_tot']:6.1f}")
        except Exception as e: row.append(f"lam{lam}: --")
      print(label,f"k={k}"," | ".join(row))
print("one-at-a-time from baseline (psi=1.1, r_f=.06, F=.03)")
for a in [0.15,0.2,0.3,0.45,0.5,0.6]: run(f"alpha={a}",alpha=a)
for E in [0.0,0.015,0.045]: run(f"E={E}",E=E)
for m in [0.0,0.01,0.03]: run(f"mu0={m}",mu0=m)
for a in [0.45,0.6]:
  for psi,rf in [(2.0,0.10),(1.5,0.10)]: run(f"alpha={a} psi={psi} rf={rf}",alpha=a,psi=psi,r_f=rf)
