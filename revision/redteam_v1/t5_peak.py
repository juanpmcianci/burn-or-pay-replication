"""Corollary 1: is log W single-peaked in Theta (rho, E_net fixed), and where is Theta^V for plausible kappa?"""
import numpy as np
rng=np.random.default_rng(1)
N=4000; ok=0; multi=0; nopeak_in_D=0; res=[]
for i in range(N):
    Th=rng.uniform(.27,.6); rho=rng.uniform(.03,.10); Es=rng.uniform(.2,.8); psi=np.exp(rng.uniform(np.log(1.05),np.log(12)))
    a=rng.uniform(.15,.6); rf=rng.uniform(.04,.15); c=rng.uniform(0,.02); mu0=rng.uniform(0,.04); muS=rng.uniform(-.02,.02)
    E=Es*rho; F=rho-E; Enet=E  # omega=0
    k=(rho/Th-c)/psi; u=Th/((1-Th)*k**(1/(psi-1)))
    L=lambda t: t**psi*((1-t)*u)**(1-psi)
    G0=rho-c*Th-L(Th); D0=rf-mu0+Enet-muS-G0
    if D0<=0: continue
    ok+=1
    kstar=rho/Th+(G0+a*D0)/(1-Th)
    for kap in [0.5*kstar, kstar, 1.5*kstar]:
        t=np.linspace(0.01,0.99,2000)
        D=rf-(mu0+kap*(t-Th))+Enet-muS-(rho-c*t-L(t))
        lw=np.where(D>0, np.log(1-t)-np.log(np.where(D>0,D,1))/a, np.nan)
        d=np.diff(lw); s=np.sign(d[np.isfinite(d)])
        ch=np.sum(s[1:]!=s[:-1])
        # also: is D>0 on whole grid? if not, logW unbounded => no interior max
        if np.any(D<=0): nopeak_in_D+=1
        elif ch>1: multi+=1
        if kap==1.5*kstar and np.all(D>0):
            res.append(t[np.nanargmax(lw)]-Th)
print("priced",ok,"cases",3*ok,"D<=0 somewhere on (0,1) (log W unbounded in Theta):",nopeak_in_D,"multi-peaked:",multi)
r=np.array(res); print("kappa=1.5 kstar: Theta^V - Theta quantiles", np.round(np.quantile(r,[.05,.5,.95]),3), "n",len(r))
