"""Check v5, task 2: lower-boundary sensitivity of the Table 2 hurdles.
Table 2 values come from riskprem2.py ('discount' rows): centred2.gap2 (Gordon value at the lower end if D0>0, power-law
Robin condition w_z=(gamma0-1)w if D0<=0<R0), domain [-7,7], 801 initial nodes, tol 1e-8.
Here: x_lo in {-7,-9,-12}, x_hi=7, node spacing held at 14/800 (n=801, 915, 1087) and doubled (n=1601,1829,2173).
Also, for comparison, centred.solve_c (Gordon value at both ends regardless of sign; used by prop2/feespeed) where it runs."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from centred import calibrated, theta_of_rho, rho_of, gamma, hurdle
from centred2 import gap2
from feasible import x_at_theta

ROWS = [(2.34, .10), (2.34, .20), (2.34, .30), (1.1, .06), (1.1, .10), (1.1, .20), (1.1, .30), (1.5, .10), (1.5, .20),
        (2.0, .10), (2.0, .20), (11.2, .15), (11.2, .20), (11.2, .30)]

def P(psi, rf): return calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0, s=0.75)

def hur(p, x, kind, **kw):
    g0 = gap2(p, x, kind, 0.0, **kw); g2 = gap2(p, x, kind, 0.2, **kw); return -g0 / ((g2 - g0) / 0.2)

def ends(p, z):
    z = np.array([z]); th = theta_of_rho(rho_of(z, p), p); G = gamma(rho_of(z, p), th, p)
    R = p.r_f - p.mu_S + p.E - G; return (R - p.mu0)[0], R[0]

def job(a):
    psi, rf, xlo, dens, solver = a
    p = P(psi, rf); x = x_at_theta(p); n = int(round((7 - xlo) / (14 / 800) * dens)) + 1
    D0, R0 = ends(p, xlo)
    out = dict(psi=psi, rf=rf, x_lo=xlo, n=n, solver=solver, D0=D0, R0=R0)
    try:
        if solver == "c2":
            out["kE"] = hur(p, x, "U", x_lo=xlo, n=n); out["kL"] = hur(p, x, "F", x_lo=xlo, n=n)
        else:
            out["kE"] = hurdle(p, x, "U", x_lo=xlo, n=n)[0]; out["kL"] = hurdle(p, x, "F", x_lo=xlo, n=n)[0]
        out["status"] = "ok"
    except Exception as e:
        out["status"] = str(e)[:50]
    return out

if __name__ == "__main__":
    jobs = [(psi, rf, xlo, dens, "c2") for psi, rf in ROWS for xlo in (-7, -9, -12) for dens in (1, 2)]
    jobs += [(psi, rf, xlo, 1, "c") for psi, rf in [(2.34, .10), (11.2, .15)] for xlo in (-7, -9, -12)]
    with Pool(12) as pool: d = pd.DataFrame(pool.map(job, jobs))
    pd.set_option("display.width", 250)
    full = d.to_string(index=False, float_format=lambda v: f"{v:.5f}")
    c2 = d[(d.solver == "c2")]
    wide = c2.pivot_table(index=["psi", "rf", "n"], columns="x_lo", values=["kE", "kL"]) if False else None
    lines = []
    for dens in (1, 2):
        sub = c2[c2.n.isin([int(round((7 - xl) / (14 / 800) * dens)) + 1 for xl in (-7, -9, -12)])]
        t = sub.pivot_table(index=["psi", "rf"], columns="x_lo", values=["kE", "kL"])
        t[("dE", "-9-(-7)")] = t[("kE", -9)] - t[("kE", -7)]; t[("dE", "-12-(-7)")] = t[("kE", -12)] - t[("kE", -7)]
        t[("dL", "-9-(-7)")] = t[("kL", -9)] - t[("kL", -7)]; t[("dL", "-12-(-7)")] = t[("kL", -12)] - t[("kL", -7)]
        lines.append(f"\n=== centred2 (paper solver), node density x{dens} ===\n" + t.to_string(float_format=lambda v: f"{v:.5f}"))
        lines.append("max |change| -7 -> -9 : kE %.5f  kL %.5f" % (t[("dE", "-9-(-7)")].abs().max(), t[("dL", "-9-(-7)")].abs().max()))
        lines.append("max |change| -7 -> -12: kE %.5f  kL %.5f" % (t[("dE", "-12-(-7)")].abs().max(), t[("dL", "-12-(-7)")].abs().max()))
        r7 = t[[("kE", -7), ("kL", -7)]].round(3).values; r12 = t[[("kE", -12), ("kL", -12)]].round(3).values
        lines.append("3-dp entries changed by -12: %d of %d" % ((r7 != r12).sum(), r7.size))
    s = full + "\n" + "\n".join(lines)
    print(s); open("output/check_v5_domain.txt", "w").write(__doc__ + "\n" + s + "\n")
