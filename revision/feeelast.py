"""Hurdles against the elasticity of dollar fee revenue with respect to market capitalisation at the reference
state, eps = d log(F W) / d log W along productivity = 1 + (d log F/dz) / (d log W/dz). Reference at F = F_max/4."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from centred import *
from feasible import x_at_theta
from calibB import closed_form

def job(a):
    name, psi, rf, s = a
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0, s=s)
    try:
        x = x_at_theta(p); lw = solve_c(p, p); h = 1e-3
        q = (lw(x + h)[0] - lw(x - h)[0]) / (2 * h)
        dlF = (np.log(fee(np.array([x + h]), p)) - np.log(fee(np.array([x - h]), p)))[0] / (2 * h)
        kL = hurdle(p, x, "F")[0]; kE = hurdle(p, x, "U")[0]
        return dict(case=name, psi=psi, rf=rf, s=s, q=q, dlogF=dlF, eps=1 + dlF / q, kss=closed_form(psi, rf), kE=kE, kL=kL)
    except Exception as e:
        return dict(case=name, psi=psi, rf=rf, s=s, err=str(e)[:60])

if __name__ == "__main__":
    S = [0.15, 0.25, 0.4, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    jobs = [(n, psi, rf, s) for n, psi, rf in [("A", 1.1, 0.10), ("A", 1.5, 0.20), ("B", 11.2, 0.20)] for s in S]
    with Pool(8) as pool:
        d = pd.DataFrame(pool.map(job, jobs))
    d["L/E"] = d.kL / d.kE; d["L/ss"] = d.kL / d.kss; d["E/ss"] = d.kE / d.kss
    d["halving"] = np.exp(d.s * np.log(7 / 3))
    d.to_csv("output/feeelast.csv", index=False)
    pd.set_option("display.width", 220); print(d.round(3).to_string(index=False))
