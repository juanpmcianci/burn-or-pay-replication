import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np, hetero
from hetero import *
SHIFT=[0.0]
orig=hetero.E_net
hetero.E_net=lambda p: orig(p)+SHIFT[0]
def logM(p,s):
    SHIFT[0]=s; M=equilibrium(p)['M']; SHIFT[0]=0; return np.log(M)
h=1e-5
print("chi* where pay ties burn if burn is capitalised only chi:  gap(chi) = G_M - (1-chi)*dlogM/dEnet")
for psi,rf,k,lam in [(1.1,.06,0,.75),(1.1,.06,.1,.75),(1.1,.10,0,.75),(2.0,.10,0,.75),(2.0,.10,.1,.5),(1.5,.10,0,1.0)]:
    p=calibrate(P2(psi=psi,r_f=rf,kappa=k),staker_wealth=lam)
    dEn=(logM(p,h)-logM(p,-h))/(2*h)
    up,dn=with_(p,E=p.E+h),with_(p,E=p.E-h)
    GM=(np.log(equilibrium(up)['M'])-np.log(equilibrium(dn)['M']))/(2*h)
    chistar=1-GM/dEn
    print(f"psi={psi} rf={rf} k={k} lam={lam}: G_M={GM:.1f} dlogM/dEnet={dEn:.1f} dlogM/drho={GM-dEn:.1f}  chi*={chistar:.2f}")
