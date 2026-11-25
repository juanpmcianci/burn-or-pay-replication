import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np
from hetero import *
print("== kappa sweep: find conflict band G_s>0>G_M ==")
for r_f,psi in [(0.06,1.1),(0.10,1.1),(0.10,2.0),(0.06,2.0)]:
  for lam in [0.45,0.5,0.75]:
    for F in [0.015,0.03]:
      ks=np.round(np.arange(0,0.61,0.01),3); res=[]
      for k in ks:
        try:
          p=calibrate(P2(psi=psi,F=F,kappa=k,r_f=r_f),staker_wealth=lam); g=gaps(p)
          res.append((k,g['G_M'],g['G_s'],g['G_h'],g['G_tot']))
        except Exception as e: res.append((k,np.nan,np.nan,np.nan,np.nan))
      a=np.array(res); ok=np.isfinite(a[:,1])
      conf=a[ok&(a[:,2]>0)&(a[:,1]<0)]; mpos=a[ok&(a[:,1]>0)]; hpos=a[ok&(a[:,3]>0)]
      kmax=a[ok,0].max()
      print(f"rf={r_f} psi={psi} lam={lam} F={F}: kmax_feasible={kmax:.2f} conflict k={conf[:,0].min() if len(conf) else None}-{conf[:,0].max() if len(conf) else None} | G_M>0 k from {mpos[:,0].min() if len(mpos) else None} | holders>0 from {hpos[:,0].min() if len(hpos) else None}")
