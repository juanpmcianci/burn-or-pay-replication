"""Lifecycle gaps and hurdles with base-path-centred feedback (centred.py), plus a lower-boundary check."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from centred import *
from feasible import x_at_theta, dtheta_drho, M_of
from calibB import closed_form

def job(a):
    psi, rf = a
    p = calibrated(beta=psi, kappa=0.0, F_max=0.12, F_min=0.0, r_f=rf)
    out = dict(psi=psi, rf=rf)
    try:
        x = x_at_theta(p)
        th0 = lambda z: theta_of_rho(rho_of(z, p), p)
        out["dth"] = dtheta_drho(p, x)[0]
        out["kss"] = closed_form(psi, rf)
        kL, gL, sL = hurdle(p, x, "F"); kE, gE, sE = hurdle(p, x, "U")
        out.update(kL=kL, G=gL, S=sL, kE=kE, GE=gE, SE=sE)
        # channels of burn-share gap at kappa=0
        out["rent"] = gap(p, x, "F", kappa=0.0, pin={"net": th0, "fb": th0}) / out["dth"]
        out["net"] = gap(p, x, "F", kappa=0.0, pin={"gamma": th0, "fb": th0}) / out["dth"]
        out["fbS"] = (gap(p, x, "F", kappa=0.2, pin={"gamma": th0, "net": th0}) - gap(p, x, "F", kappa=0.0, pin={"gamma": th0, "net": th0})) / 0.2 / out["dth"]
        out["M"] = M_of(p, x)[0]
        kL9 = hurdle(p, x, "F", x_lo=-9.0)[0]; out["kL_xlo9"] = kL9
        kLa = hurdle(p, x, "F", k1=0.4)[0]; out["kL_k1"] = kLa
        out["status"] = "ok"
    except Exception as e:
        out["status"] = str(e)[:60]
    return out

if __name__ == "__main__":
    cases = [(psi, rf) for psi in [1.1, 1.5, 2.0] for rf in [0.06, 0.10, 0.15, 0.20, 0.30]] + [(11.2, rf) for rf in [0.15, 0.20, 0.30]]
    with Pool(8) as pool:
        d = pd.DataFrame(pool.map(job, cases))
    d.to_csv("output/table_centred.csv", index=False)
    pd.set_option("display.width", 250)
    ok = d[d.status == "ok"].copy()
    ok["L/ss"] = ok.kL / ok.kss; ok["E/ss"] = ok.kE / ok.kss; ok["L/E"] = ok.kL / ok.kE; ok["kL/r"] = ok.kL / 0.1333
    print(ok[["psi","rf","dth","GE","G","SE","S","rent","net","fbS","M","kss","kE","kL","L/ss","E/ss","L/E","kL/r","kL_xlo9","kL_k1"]].round(3).to_string(index=False))
    print(d[d.status != "ok"][["psi","rf","status"]].to_string(index=False))
