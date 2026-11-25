"""Check v5, task 1: fee-level sensitivity of the hurdle ratio kappa*_omega / kappa*_E.
Paper's Table 2 setting (riskprem2.py 'discount', centred2.gap2, domain [-7,7], 801 nodes):
  calibrated(beta=psi, r_f, F_max=0.12, F_min=0, s=0.75, x0=0, E=0.03, c=0.01, mu0=0.02, alpha=0.3, sigma_A=0.35),
  u = u_for(0.45, 0.06, psi, c) so Theta=0.45 at rho=0.06; reference z0 solves Theta(rho(z0))=0.45, i.e. F(z0)=0.06-E.
Variants: F_ref in {0.03, 0.01, 0.005}; E = 0.06 - F_ref; F_max = 4 F_ref (so F(z0)=F_max/4, z0 = s ln 3 unchanged);
F_min=0, s=0.75, u unchanged (depends only on Theta=0.45, rho=0.06, psi, c). Hurdles: -0.2 G(0)/(G(0.2)-G(0))."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from centred import calibrated, fee
from centred2 import gap2
from feasible import x_at_theta
from calibB import closed_form

def hur(p, x, kind, **kw):
    g0 = gap2(p, x, kind, 0.0, **kw); g2 = gap2(p, x, kind, 0.2, **kw); return -g0 / ((g2 - g0) / 0.2), g0

def job(a):
    psi, rf, Fr = a
    E = 0.06 - Fr
    p = calibrated(beta=psi, kappa=0.0, r_f=rf, E=E, F_max=4 * Fr, F_min=0.0, s=0.75)
    x = x_at_theta(p)
    out = dict(psi=psi, rf=rf, F_ref=Fr, E=E, F_max=4 * Fr, z0=x, F_at_z0=fee(np.array([x]), p)[0], u=p.u,
               kss=closed_form(psi, rf, E=E))
    try:
        out["kE"], out["G0E"] = hur(p, x, "U"); out["kL"], out["G0L"] = hur(p, x, "F"); out["status"] = "ok"
    except Exception as e:
        out["status"] = str(e)[:60]
    return out

if __name__ == "__main__":
    cells = [(psi, 0.20, Fr) for psi in (2.34, 11.2) for Fr in (0.03, 0.01, 0.005)]
    cells += [(2.34, 0.10, Fr) for Fr in (0.03, 0.01, 0.005)] + [(11.2, 0.15, Fr) for Fr in (0.03, 0.01, 0.005)]
    with Pool(12) as pool: d = pd.DataFrame(pool.map(job, cells))
    d["ratio_L_E"] = d.kL / d.kE; d["kL_kss"] = d.kL / d.kss; d["kE_kss"] = d.kE / d.kss
    pd.set_option("display.width", 250)
    s = d.to_string(index=False, float_format=lambda v: f"{v:.5f}")
    print(s); open("output/check_v5_feelevel.txt", "w").write(__doc__ + "\n" + s + "\n")
