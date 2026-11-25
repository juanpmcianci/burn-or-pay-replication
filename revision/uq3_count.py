import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "uq"))
from uq2 import setup
from centred import theta_of_rho, rho_of, gamma, fee, with_
sub = pd.read_parquet(Path(__file__).resolve().parents[1] / "uq/output/uq0_draws.parquet").sample(400, random_state=23)
d = pd.read_parquet("output/uq2_draws.parquet")
rows = []
for cfg in ["QD", "Q75"]:
    for i, row in sub.iterrows():
        p, s, sig = setup(row, cfg, i)
        out = {}
        for name, Fv in [("lo", p.F_max), ("hi", p.F_min), ("m7", None)]:
            x = np.array([-7.0]) if Fv is None else None
            q = p if Fv is None else with_(p, F_max=Fv, F_min=Fv)
            xx = np.array([-7.0]) if Fv is None else np.array([0.0])
            th = theta_of_rho(rho_of(xx, q), q); G = gamma(rho_of(xx, q), th, q)
            R = q.r_f - q.mu_S + q.E - G
            out[name + "_R"] = R[0]; out[name + "_D"] = R[0] - q.mu0
        rows.append(dict(cfg=cfg, i=i, **out))
c = pd.DataFrame(rows).merge(d[["cfg", "i", "status"]], on=["cfg", "i"])
for cfg, g in c.groupby("cfg"):
    print(cfg, g.status.value_counts().to_dict())
    np_ = g[g.status == "no price"]
    print("  no-price: upper D<=0", (np_.hi_D <= 0).sum(), "| lower-limit R<=0", (np_.lo_R <= 0).sum(),
          "| R(-7)<=0", (np_.m7_R <= 0).sum(), "| recoverable (D_hi>0, R_lo>0)", ((np_.hi_D > 0) & (np_.lo_R > 0)).sum())
    ok = g[g.status == "ok"]
    print("  ok draws with lower-limit R<=0:", (ok.lo_R <= 0).sum(), " lower-limit D<=0:", (ok.lo_D <= 0).sum())
c.to_csv("output/uq3_condition.csv", index=False)
