"""RT v2 check: (a) local steady-state hurdle kappa*(z) along the baseline path; (b) fee-yield and dollar-fee
elasticity w.r.t. productivity at the reference state for each fee speed s; (c) hurdles at s=0.15."""
import sys, numpy as np, pandas as pd
R=__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"..")
sys.path.insert(0,R); sys.path.insert(0,R+"/../gonogo")
from centred import *
from feasible import x_at_theta
from multiprocessing import Pool

def local_kstar(p, z):
    z=np.atleast_1d(z); rho=rho_of(z,p); th=theta_of_rho(rho,p); G=gamma(rho,th,p)
    D=p.r_f-p.mu0+p.E-p.mu_S-G; r=rho/th
    return r+G/(1-th)+p.alpha*D/(1-th), th, D, 1/(p.alpha*D)

def job(a):
    psi,rf,s=a
    p=calibrated(beta=psi,kappa=0.0,r_f=rf,F_max=0.12,F_min=0.0,s=s)
    x=x_at_theta(p); base=p
    lw=solve_c(p,base)
    h=1e-3; dlogW=(lw(x+h)[0]-lw(x-h)[0])/(2*h)
    dlogF=(np.log(fee(np.array([x+h]),p))-np.log(fee(np.array([x-h]),p)))[0]/(2*h)
    out=dict(psi=psi,rf=rf,s=s,x=x,dlogF_dz=dlogF,dlogW_dz=dlogW,dollar_fee_elast=(dlogF+dlogW)/dlogW)
    if s in (0.15,):
        out["kL"]=hurdle(p,x,"F")[0]; out["kE"]=hurdle(p,x,"U")[0]
    return out

if __name__=="__main__":
    for psi,rf in [(1.5,0.2),(11.2,0.2)]:
        p=calibrated(beta=psi,kappa=0.0,r_f=rf,F_max=0.12,F_min=0.0); x=x_at_theta(p)
        zs=x+np.array([-4,-2,-1,-0.5,0,0.5,1,2,4])
        k,th,D,M=local_kstar(p,zs)
        print(f"psi={psi} rf={rf} ref x={x:.3f}")
        print(pd.DataFrame(dict(dz=zs-x,F=fee(zs,p),Theta=th,dth=[np.nan]*len(zs),kstar_local=k,D=D,M=M)).drop(columns="dth").round(3).to_string(index=False))
    jobs=[(psi,rf,s) for psi,rf in [(1.5,0.2),(11.2,0.2)] for s in [0.15,0.25,0.4,0.75,1.5,3.0]]
    with Pool(4) as pool: res=pool.map(job,jobs)
    d=pd.DataFrame(res); d["halving"]=np.exp(d.s*np.log(7/3))
    if "kL" in d: d["L/E"]=d.kL/d.kE
    pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
