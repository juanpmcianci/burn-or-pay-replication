"""Red team v4: domain sensitivity for draws whose lower end has D < 0 < R (power-law boundary). Re-solve 24 such
priced QD/Q75 draws with the lower boundary pushed 4 s further and 1801 nodes; compare hurdles."""
import numpy as np, pandas as pd
from multiprocessing import Pool
import uq3
from uq2 import setup
from centred import theta_of_rho, rho_of, gamma, with_
from centred2 import gap2
def lowD(p):
    q = with_(p, F_max=p.F_max, F_min=p.F_max); x = np.array([0.0]); th = theta_of_rho(rho_of(x, q), q)
    G = gamma(rho_of(x, q), th, q); R = q.r_f - q.mu_S + q.E - G; return (R - q.mu0)[0]
def job(a):
    i, row, cfg, kF0 = a
    p, s, sig = setup(row, cfg, i)
    kw = dict(x_lo=min(-7.0, p.x0 - 14 * p.s), x_hi=max(7.0, p.x0 + 10 * p.s), n=1801)
    try:
        g0 = gap2(p, 0.0, "F", 0.0, **kw); g2 = gap2(p, 0.0, "F", 0.2, **kw); k = -g0 / ((g2 - g0) / 0.2)
    except Exception as e: k = np.nan
    return dict(i=i, cfg=cfg, kF=kF0, kF_wide=k)
if __name__ == "__main__":
    o = pd.read_parquet("output/uq3_draws.parquet"); sub = uq3.draws()
    o = o[(o.status == "ok") & o.kF.notna()]
    o["Dlo"] = [lowD(setup(sub.loc[r.i], r.cfg, r.i)[0]) for _, r in o.iterrows()]
    neg = o[o.Dlo < 0]; print("priced draws with D<0 at lower end:", neg.groupby("cfg").size().to_dict())
    pick = pd.concat([g.sample(min(12, len(g)), random_state=1) for _, g in neg.groupby("cfg")])
    with Pool(10) as pool: res = pool.map(job, [(r.i, sub.loc[r.i], r.cfg, r.kF) for _, r in pick.iterrows()])
    d = pd.DataFrame(res); d["diff"] = d.kF_wide - d.kF; d["rel"] = d["diff"] / d.kF
    d.to_csv("output/domain_check.csv", index=False); print(d.round(4).to_string(index=False))
    print("max |rel diff|", d.rel.abs().max())
