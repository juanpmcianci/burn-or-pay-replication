"""Revision A1: Table 2 recomputed with the feasible derivative. Same grid as draft 0 (gonogo/sweep.py)."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from feasible import *

def row(args):
    psi, rf, k = args
    p = calibrated(beta=psi, kappa=k, F_max=0.12, F_min=0.0, r_f=rf)
    out = dict(psi=psi, rf=rf, kappa=k)
    try:
        x = x_at_theta(p)
        out["x"] = x
        out["dth"] = dtheta_drho(p, x)[0]
        out["G_U"] = G_uniform(p, x)[0]
        out["G_F"] = G_feasible(p, x)[0]
        chF = channels(p, x, "feasible"); chU = channels(p, x, "uniform")
        for c in CH:
            out[f"F_{c}"] = chF[c][0]; out[f"U_{c}"] = chU[c][0]
        out["M"] = M_of(p, x)[0]
        out["status"] = "ok"
    except Exception as e:
        out["status"] = f"no price: {str(e)[:60]}"
    return out

if __name__ == "__main__":
    grid = [(psi, rf, k) for psi in [1.1, 1.5, 2.0] for rf in [0.06, 0.10] for k in [0.0, 0.1, 0.2, 0.3]]
    with Pool(8) as pool:
        res = pool.map(row, grid)
    d = pd.DataFrame(res)
    d.to_csv("output/table2_feasible.csv", index=False)
    ok = d[d.status == "ok"].copy()
    ok["V_F"] = ok.G_F / ok.dth; ok["V_U"] = ok.G_U / ok.dth
    for c in CH:
        ok[f"VF_{c}"] = ok[f"F_{c}"] / ok.dth
    ok["F_resid"] = ok.G_F - ok[[f"F_{c}" for c in CH]].sum(axis=1)
    pd.set_option("display.width", 250)
    print(ok[["psi", "rf", "kappa", "dth", "G_U", "G_F", "V_U", "V_F", "VF_gamma", "VF_net", "VF_fb", "F_resid", "M"]].round(3).to_string(index=False))
    print(d[d.status != "ok"][["psi", "rf", "kappa", "status"]].to_string(index=False))
