"""RT v2 check M-a: why kL > kE under centred feedback. Channel-by-channel fading ratio (burn share / issuance)."""
import sys, numpy as np
R=__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"..")
sys.path.insert(0,R); sys.path.insert(0,R+"/../gonogo")
import centred as C
from centred import *
from feasible import x_at_theta
from multiprocessing import Pool
from scipy.integrate import solve_bvp

def solve_m(p, base, freeze_motion=False, x_lo=-7.0, x_hi=7.0, n=801, tol=1e-8):
    s2 = p.sigma_A**2
    th0 = lambda z: theta_of_rho(rho_of(z, base), base)
    def parts(x):
        th = theta_of_rho(rho_of(x, p), p)
        mA = p.mu0 + p.kappa*(th - th0(x))
        G = gamma(rho_of(x,p), th, p)
        En = p.E - p.w_B*fee(x,p)
        return mA, G, En, th
    def rhs(x,y):
        w, wx = y; w=np.maximum(w,1e-12)
        mA,G,En,tn = parts(x)
        mm = p.mu0 if freeze_motion else mA
        flow=(mA+G-En+p.mu_S-p.r_f)*w+(1-p.alpha)*(1-tn)**p.alpha*w**(1-p.alpha)
        return np.vstack([wx, -(2.0/s2)*((mm+0.5*s2)*wx+flow)])
    def gw(x):
        mA,G,En,tn=parts(x); den=p.r_f-(mA-En+p.mu_S)-G
        return (1-tn)*((1-p.alpha)/den)**(1/p.alpha)
    xs=np.linspace(x_lo,x_hi,n); w0=gw(xs)
    sol=solve_bvp(rhs, lambda a,b: np.array([a[0]-gw(np.array([x_lo]))[0], b[0]-gw(np.array([x_hi]))[0]]), xs, np.vstack([w0,np.gradient(w0,xs)]), tol=tol, max_nodes=200000)
    return lambda z: np.log(sol.sol(np.atleast_1d(z))[0])+np.atleast_1d(z)

def gapm(p,x,kind,kappa,fm):
    base=with_(p,kappa=0.0); q=with_(p,kappa=kappa)
    if kind=="F":
        up=solve_m(with_(q,w_B=q.w_B+C.DW),base,fm)(x); dn=solve_m(with_(q,w_B=q.w_B-C.DW),base,fm)(x)
        return (-(up-dn)/(2*C.DW)/fee(np.atleast_1d(x),q))[0]
    up=solve_m(with_(q,E=q.E+C.H),base,fm)(x); dn=solve_m(with_(q,E=q.E-C.H),base,fm)(x)
    return ((up-dn)/(2*C.H))[0]

def job(a):
    psi,rf,kw=a
    p=calibrated(beta=psi,kappa=0.0,r_f=rf,**{**dict(F_max=0.12,F_min=0.0),**kw})
    x=x_at_theta(p); base=with_(p,kappa=0.0)
    th0=lambda z: theta_of_rho(rho_of(z,base),base)
    out=dict(psi=psi,rf=rf,**kw)
    for kind in "FU":
        g0=gap(p,x,kind,kappa=0.0); s=(gap(p,x,kind,kappa=0.2)-g0)/0.2
        # channels at kappa=0: respond only in gamma (pin net), only in network (pin gamma)
        rent=gap(p,x,kind,kappa=0.0,pin={"net":th0}); net=gap(p,x,kind,kappa=0.0,pin={"gamma":th0})
        sfm=(gapm(p,x,kind,0.2,True)-gapm(p,x,kind,0.0,True))/0.2
        out.update({f"G0{kind}":g0,f"S{kind}":s,f"rent{kind}":rent,f"net{kind}":net,f"Sfm{kind}":sfm})
    return out

if __name__=="__main__":
    jobs=[(1.1,0.10,{}),(1.5,0.20,{}),(11.2,0.20,{}),(1.5,0.20,dict(s=0.25)),(11.2,0.20,dict(s=0.25))]
    with Pool(4) as pool: res=pool.map(job,jobs)
    import pandas as pd
    d=pd.DataFrame(res)
    for c in ["G0","S","rent","net","Sfm"]: d[c+"_ratio"]=d[c+"F"]/d[c+"U"]
    d["kL"]=-d.G0F/d.SF; d["kE"]=-d.G0U/d.SU; d["kL_fm"]=-d.G0F/d.SfmF; d["kE_fm"]=-d.G0U/d.SfmU
    pd.set_option("display.width",250)
    print(d.round(3).to_string(index=False))
