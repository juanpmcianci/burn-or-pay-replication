"""Hurdles (centred feedback) as a function of how fast fees fall (logistic width s), and other fee-schedule
variants. Halving distance from the reference state (F = F_max/4): productivity must rise by exp(s ln(7/3))."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from centred import *
from feasible import x_at_theta
from calibB import closed_form

def job(a):
    name, psi, rf, kw = a
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, **{**dict(F_max=0.12, F_min=0.0), **kw})
    try:
        x = x_at_theta(p)
        kL = hurdle(p, x, "F")[0]; kE = hurdle(p, x, "U")[0]
        return dict(case=name, psi=psi, rf=rf, **kw, F_ref=float(fee(np.array([x]), p)[0]), kss=closed_form(psi, rf), kE=kE, kL=kL)
    except Exception as e:
        return dict(case=name, psi=psi, rf=rf, **kw, err=str(e)[:50])

if __name__ == "__main__":
    jobs = []
    for name, psi, rf in [("A", 1.1, 0.10), ("A", 1.5, 0.20), ("B", 11.2, 0.20)]:
        for s in [0.25, 0.4, 0.75, 1.5, 3.0]:
            jobs.append((name, psi, rf, dict(s=s)))
        jobs.append((name, psi, rf, dict(F_min=0.01)))
        jobs.append((name, psi, rf, dict(F_max=0.06)))
    with Pool(8) as pool:
        d = pd.DataFrame(pool.map(job, jobs))
    d["L/E"] = d.kL / d.kE; d["L/ss"] = d.kL / d.kss; d["E/ss"] = d.kE / d.kss
    d["halving_factor"] = np.exp(d.get("s", 0.75).fillna(0.75) * np.log(7 / 3))
    d.to_csv("output/feespeed.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.round(3).to_string(index=False))
