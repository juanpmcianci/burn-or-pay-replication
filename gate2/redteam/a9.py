import sys; sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
import numpy as np
from hetero import *
# check: sum_j Omega_j == PV(real flows) = [conv + F*M - c*Theta*M - Lock]/(r_f - mu_A - mu_S)  (no double count)
for kw,lam in [(dict(),0.75),(dict(kappa=0.1,psi=2.0,r_f=0.1),0.5),(dict(w_B=0.5),0.6)]:
    p=calibrate(P2(**kw),staker_wealth=lam); eq=equilibrium(p); Th=eq['Theta']; N=1-Th; M=eq['M']
    e={j:v['m']*v['w']/M for j,v in eq['types'].items()}; W=welfare(p,eq,e)
    tot=sum(v['total'] for v in W.values())
    conv=sum(v['m']*v['w']**(1-p.alpha)*N**p.alpha for v in eq['types'].values())
    lock=0
    for j,v in eq['types'].items():
        u=p.u_s if j=='s' else p.u_h
        if v['theta']>0: lock+=v['m']*v['w']*v['theta']**p.psi*(N*u)**(1-p.psi)
    real=(conv+p.F*M-p.c*Th*M-lock)/(p.r_f-mu_A(p,Th)-p.mu_S)
    print(kw,lam,"sum Omega",round(tot,6),"PV real flows",round(real,6))
