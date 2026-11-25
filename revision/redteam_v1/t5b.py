import numpy as np
rng=np.random.default_rng(2)
out={0.0:[],0.5:[],1.5:[]}
for i in range(4000):
    Th=rng.uniform(.27,.6); rho=rng.uniform(.03,.10); Es=rng.uniform(.2,.8); psi=np.exp(rng.uniform(np.log(1.05),np.log(12)))
    a=rng.uniform(.15,.6); rf=rng.uniform(.04,.15); c=rng.uniform(0,.02); mu0=rng.uniform(0,.04); muS=rng.uniform(-.02,.02)
    E=Es*rho; k=(rho/Th-c)/psi; u=Th/((1-Th)*k**(1/(psi-1)))
    L=lambda t: t**psi*((1-t)*u)**(1-psi)
    G0=rho-c*Th-L(Th); D0=rf-mu0+E-muS-G0
    if D0<=0: continue
    ks=rho/Th+(G0+a*D0)/(1-Th)
    for m in out:
        kap=m*ks; t=np.linspace(0.01,0.99,4000)
        D=rf-(mu0+kap*(t-Th))+E-muS-(rho-c*t-L(t))
        if np.any(D<=0): continue
        lw=np.log(1-t)-np.log(D)/a; l0=np.log(1-Th)-np.log(D0)/a
        out[m].append((t[np.argmax(lw)]-Th, np.max(lw)-l0))
for m,v in out.items():
    v=np.array(v); print("kappa/kstar",m,"Theta^V-Theta q05/50/95",np.round(np.quantile(v[:,0],[.05,.5,.95]),3),"max gain in logW",np.round(np.quantile(v[:,1],[.05,.5,.95]),3))
