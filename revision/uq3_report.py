"""Report for uq3 (items 16-17): true price condition, wide domain, 1600 draws, with Monte Carlo errors."""
import numpy as np, pandas as pd
o = pd.read_parquet("output/uq3_draws.parquet")
pool = pd.read_parquet("../uq/output/uq0_draws.parquet")
q = lambda x: np.round(np.quantile(x, [.05, .25, .5, .75, .95]), 2)
def share(x):
    p = np.mean(x); return round(p, 3), round(np.sqrt(p * (1 - p) / len(x)), 3)
for cfg in ["QD", "Q75"]:
    g = o[o.cfg == cfg]; N = len(g)
    print(f"=== {cfg}: N {N}", g.status.value_counts().to_dict())
    print("   no-price reasons", g[g.status == "no price"].why.str[:25].value_counts().to_dict())
    ok = g[g.status == "ok"].copy(); n = len(ok)
    ok["LR"] = ok.kF / ok.r; ok["Lss"] = ok.kF / ok.kstar; ok["Ess"] = ok.kU / ok.kstar; ok["LE"] = ok.kF / ok.kU
    for c in ["LR", "Lss", "Ess", "LE", "eps"]: print(f"  {c:4s}", q(ok[c].dropna()))
    w10 = abs(ok.Lss - 1) < 0.1; w20 = abs(ok.Lss - 1) < 0.2
    print("  within10 of priced", share(w10), "| within20", share(w20), "| kL>kE", share(ok.kF > ok.kU), "| kL>kstar", share(ok.kF > ok.kstar))
    print("  median Lss MC se (bootstrap)", round(np.std([np.median(ok.Lss.sample(n, replace=True, random_state=b)) for b in range(400)]), 3))
    print("  within10 as share of all draws", round(w10.sum() / N, 3), "| bounds (failures worst/best):", round(w10.sum() / N, 3), round((w10.sum() + (N - n)) / N, 3))
    # reweight priced draws to prior psi distribution (10 log-psi bins)
    bins = np.quantile(np.log(pool.psi), np.linspace(0, 1, 11)); bins[0] -= 1e-9; bins[-1] += 1e-9
    b_ok = np.digitize(np.log(ok.psi), bins); wts = pd.Series(b_ok).map(1 / pd.Series(b_ok).value_counts()).values
    print("  within10 reweighted to prior psi", round(np.sum(wts * w10.values) / np.sum(wts), 3))
    for nm, m in [("psi<2", ok.psi < 2), ("psi>=2", ok.psi >= 2)]:
        print(f"  {nm} n={m.sum()} Lss", q(ok.Lss[m].dropna()))
    for bt in (0, 1):
        m = ok.batch == bt; print(f"  batch {bt}: n {m.sum()} median Lss {ok.Lss[m].median():.3f} within10 {np.mean(w10[m]):.3f}")
    print("  bound kL > r + G/(1-Th):", np.mean(ok.kF > ok.r + ok.Gamma / (1 - ok.Theta)))
    print("  no-price share by psi:", g.assign(np_=g.status == "no price").groupby(pd.cut(g.psi, [1, 1.5, 2, 3, 5, 20])).np_.mean().round(2).to_dict())
