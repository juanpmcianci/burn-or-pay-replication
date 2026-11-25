import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np, itertools
from hetero import *
print("sign map at psi=1.1 rf=.06 F=.03 lam=.75 E=.03 c=.01: code M/s/h  (+ payout preferred)")
for k in [0.05,0.1,0.15]:
  print("kappa",k,"rows alpha, cols mu0 0.00 .01 .02 .03 .035 .04")
  for a in [0.3,0.4,0.5,0.6,0.7]:
    s=[]
    for m in [0,0.01,0.02,0.03,0.035,0.04]:
      try:
        g=gaps(calibrate(P2(alpha=a,mu0=m,kappa=k),staker_wealth=0.75)); s.append(''.join('+' if g[x]>0 else '-' for x in ['G_M','G_s','G_h']) + ('T' if g['G_tot']>0 else 't'))
      except Exception: s.append(' -- ')
    print(f"  a={a}: "," ".join(s))
print("\nPLAN box (+-50% lam, psi, F) around alpha=0.6, mu0=0.03, kappa=0.1:")
n=c1=c2=0
for lam,psi,F in itertools.product([0.45,0.5,0.6,0.75,0.9,1.0],[1.05,1.1,1.25,1.5,1.65],[0.015,0.02,0.03,0.04,0.045]):
    try: g=gaps(calibrate(P2(alpha=0.6,mu0=0.03,kappa=0.1,psi=psi,F=F),staker_wealth=lam))
    except Exception: continue
    n+=1; c1+=(g['G_s']>0>g['G_M']); c2+=(np.sign(g['G_tot'])!=np.sign(g['G_M']))
print(n,"feasible; staker conflict",c1,"; total vs M disagree",c2)
