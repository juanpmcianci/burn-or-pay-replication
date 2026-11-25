"""Red team v4: risk premium in the drift across the Table 5 grid. Discount reading: (r_f, mu0 = 0.02).
Drift reading: (0.06, mu0 = 0.02 - (r_f - 0.06)), same steady state. Uses centred2 (paper's price condition)."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from centred import calibrated
from centred2 import gap2
from feasible import x_at_theta
def hur(p, x, kind):
    g0 = gap2(p, x, kind, 0.0); g2 = gap2(p, x, kind, 0.2); return -g0 / ((g2 - g0) / 0.2)
def job(a):
    psi, rf, mode = a
    p = calibrated(beta=psi, kappa=0.0, r_f=rf if mode == "discount" else 0.06, F_max=0.12, F_min=0.0, s=0.75,
                   mu0=0.02 if mode == "discount" else 0.02 - (rf - 0.06))
    x = x_at_theta(p); r = 0.06 / 0.45; c = 0.01; G = (psi - 1) / psi * (0.06 - c * 0.45)
    D = rf - 0.02 + 0.03 - G; ks = r + G / 0.55 + 0.3 * D / 0.55
    try: kL, kE = hur(p, x, "F"), hur(p, x, "U"); st = "ok"
    except Exception as e: kL = kE = np.nan; st = str(e)[:40]
    return dict(psi=psi, rf=rf, mode=mode, kstar=ks, kE=kE, kL=kL, status=st)
if __name__ == "__main__":
    jobs = [(psi, rf, m) for psi in [1.1, 1.5, 2.0, 2.34] for rf in [0.10, 0.20, 0.30] for m in ["discount", "drift"]]
    jobs += [(11.2, rf, m) for rf in [0.15, 0.20, 0.30] for m in ["discount", "drift"]]
    with Pool(10) as pool: res = pool.map(job, jobs)
    d = pd.DataFrame(res); d["L_ss"] = d.kL / d.kstar; d["L_E"] = d.kL / d.kE
    d.to_csv("output/riskprem2.csv", index=False); print(d.round(3).to_string(index=False))
