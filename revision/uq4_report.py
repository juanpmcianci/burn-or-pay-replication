"""Compare uq4 (risk premium in the drift, r0 = 0.04) with uq3 (risk premium as discount rate), draw by draw."""
import sys, numpy as np, pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else "output/uq4_draws.parquet"
d = pd.read_parquet(src); r = pd.read_parquet("output/uq3_draws.parquet")
pool = pd.read_parquet("../uq/output/uq0_draws.parquet")
q = lambda x: np.round(np.quantile(x, [.05, .25, .5, .75, .95]), 2)
bins = np.quantile(np.log(pool.psi), np.linspace(0, 1, 11)); bins[0] -= 1e-9; bins[-1] += 1e-9
def wmed(x, psi):
    b = np.digitize(np.log(psi), bins); w = pd.Series(b).map(1 / pd.Series(b).value_counts()).values
    o = np.argsort(x.values); cw = np.cumsum(w[o]) / w.sum(); return round(float(x.values[o][np.searchsorted(cw, .5)]), 3)
def share(x):
    p = np.mean(x); return round(p, 3), round(np.sqrt(p * (1 - p) / len(x)), 3)
for cfg in ["QD", "Q75"]:
    g = d[d.cfg == cfg]; N = len(g)
    print(f"=== {cfg} drift: N {N}", g.status.value_counts().to_dict())
    m = g.merge(r[r.cfg == cfg][["i", "status", "kF", "kU"]], on="i", suffixes=("", "_r"))
    print("   status drift x discount:", pd.crosstab(m.status.str[:8], m.status_r.str[:8]).to_dict())
    ok = g[g.status == "ok"].copy()
    ok["Lss"] = ok.kF / ok.kstar; ok["Ess"] = ok.kU / ok.kstar; ok["LE"] = ok.kF / ok.kU
    for c in ["Lss", "Ess", "LE"]: print(f"  {c:4s}", q(ok[c].dropna()), "| prior-psi-weighted median", wmed(ok[c].dropna(), ok.psi[ok[c].notna()]))
    w10 = abs(ok.Lss - 1) < 0.1
    print("  within10 of priced", share(w10), "| kL>kE", share(ok.kF > ok.kU), "| kL>kstar", share(ok.kF > ok.kstar))
    print("  within10 bounds over all draws", round(w10.sum() / N, 3), round((w10.sum() + N - len(ok)) / N, 3))
    b = m[(m.status == "ok") & (m.status_r == "ok")].copy(); print(f"  both priced n={len(b)}")
    b["dL"] = b.kF / b.kF_r; b["dE"] = b.kU / b.kU_r; b["LE_d"] = b.kF / b.kU; b["LE_r"] = b.kF_r / b.kU_r
    print("   drift/discount kL", q(b.dL.dropna()), "| kE", q(b.dE.dropna()))
    print("   share kL rises", share(b.dL > 1), "| share kE rises", share(b.dE > 1))
    print("   LE discount", q(b.LE_r.dropna()), "-> drift", q(b.LE_d.dropna()))
    b["prem"] = b.r_f - 0.04
    print("   median dL by premium:", b.groupby(pd.cut(b.prem, [-.001, .03, .06, .09, .12])).dL.median().round(3).to_dict())
    print("   median LE by premium (drift):", b.groupby(pd.cut(b.prem, [-.001, .03, .06, .09, .12])).LE_d.median().round(3).to_dict())
