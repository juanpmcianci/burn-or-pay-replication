import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np
from hetero import *
from scipy.optimize import brentq
def one_sided(p,h=1e-5):
    e0=equilibrium(p); M0=e0['M']; e={j:v['m']*v['w']/M0 for j,v in e0['types'].items()}
    W0=welfare(p,e0,e); out={}
    for s in (+1,-1):
        q=with_(p,E=p.E+s*h); eq=equilibrium(q); W=welfare(q,eq,e)
        out[s]=dict(G_M=s*(np.log(eq['M'])-np.log(M0))/h, G_s=s*(W['s']['total']-W0['s']['total'])/h/M0,
                    dTh=s*(eq['Theta']-e0['Theta'])/h, ths=eq['types']['s']['theta'])
    return out
print("== kink at lam=0.45 corner (theta_s=1 exactly) ==")
for k in [0,0.1]:
    p=calibrate(P2(kappa=k),staker_wealth=0.45); o=one_sided(p)
    print(k,{s:{a:round(b,2) for a,b in v.items()} for s,v in o.items()})
print("== strict corner: u_s large so theta_s=1 with slack; m_s solved for Theta=0.45 ==")
for psi in [1.1,2.0]:
  for rf in [0.06,0.10]:
    for k in [0,0.1]:
      for umult in [1.5,5,50]:
        base=calibrate(P2(psi=psi,r_f=rf,kappa=k),staker_wealth=0.45)
        us=base.u_s*umult
        def f(ms):
            try: return equilibrium(with_(base,u_s=us,m_s=ms))['Theta']-0.45
            except Exception: return np.nan
        try:
            ms=brentq(f,1e-6,base.m_s*1.5 if f(base.m_s*1.5)>0 else 0.999)
            p=with_(base,u_s=us,m_s=ms); g=gaps(p)
            print(f"psi={psi} rf={rf} k={k} u x{umult}: m_s={ms:.4f} theta_s={g['theta_s']:.3f} e_s={g['e_s']:.3f} dTh/drho={g['dTheta_drho']:.2f} G_M={g['G_M']:.1f} G_s={g['G_s']:.1f} (endow {g['G_s_endow']:.1f} surplus {g['G_s_surplus']:.1f}) G_h={g['G_h']:.1f}")
        except Exception as e: print(psi,rf,k,umult,"fail",str(e)[:60])
