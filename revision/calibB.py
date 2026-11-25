"""Revision B15/A1: the Table-2 calibration (psi = 11.2) with the feasible derivative, and steady-state closed-form
hurdles at the lifecycle reference state for comparison with the lifecycle tie points."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from table2 import row
from tiepoints import case

def closed_form(psi, rf, Th=0.45, rho=0.06, c=0.01, E=0.03, mu0=0.02, a=0.3):
    G = (psi - 1) / psi * (rho - c * Th); D = rf - mu0 + E - G
    return rho / Th + (G + a * D) / (1 - Th) if D > 0 else np.nan

if __name__ == "__main__":
    with Pool(4) as pool:
        rows = pool.map(row, [(11.2, 0.15, k) for k in [0.0, 0.2, 0.4]])
        tp = pool.map(case, [(11.2, rf, {}) for rf in [0.15, 0.20, 0.30]])
    d = pd.DataFrame(rows); d["V_F"] = d.G_F / d.dth
    print(d[["psi", "rf", "kappa", "dth", "G_U", "G_F", "V_F", "F_gamma", "F_net", "F_fb", "M"]].round(3).to_string(index=False))
    print(pd.DataFrame(tp).round(4).to_string(index=False))
    for psi in [1.1, 1.5, 2.0, 11.2]:
        print(psi, [round(closed_form(psi, rf), 3) for rf in [0.06, 0.10, 0.15, 0.20, 0.30]])
