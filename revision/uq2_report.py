import numpy as np, pandas as pd
o = pd.read_parquet("output/uq2_draws.parquet")
m = pd.read_parquet("output/uq1c_draws.parquet")[["status", "kF", "kU"]].rename(columns=lambda c: c + "_mid")
q = lambda s: np.round(s.dropna().quantile([.05, .25, .5, .75, .95]).values, 2)
for cfg, g in o.groupby("cfg"):
    print(f"=== {cfg}: status", g.status.value_counts().to_dict())
    ok = g[g.status == "ok"].copy()
    print(" slopes<=0 F,U:", int((ok.SF <= 0).sum()), int((ok.SU <= 0).sum()), "| max |p2-fd| F,U:", round((ok.kF_p2 - ok.kF).abs().max(), 5), round((ok.kU_p2 - ok.kU).abs().max(), 5), "| qmin<0:", int((ok.qmin < 0).sum()))
    ok["LR"] = ok.kF / ok.r; ok["Lss"] = ok.kF / ok.kstar; ok["Ess"] = ok.kU / ok.kstar; ok["LE"] = ok.kF / ok.kU
    for c in ["LR", "Lss", "Ess", "LE", "eps", "s", "sigma_A"]: print(f"  {c:7s}", q(ok[c]))
    print("  share kL>kE", round(np.mean(ok.kF > ok.kU), 3), "| kL>kstar", round(np.mean(ok.kF > ok.kstar), 3), "| |Lss-1|<0.1", round(np.mean(abs(ok.Lss - 1) < 0.1), 3), "| |Lss-1|<0.2", round(np.mean(abs(ok.Lss - 1) < 0.2), 3))
    b = ok.r + ok.Gamma / (1 - ok.Theta)
    print("  kL > r+G/(1-Th):", round(np.mean(ok.kF > b), 3), "min margin", round(float((ok.kF - b).min()), 4), "| kE > bound", round(np.mean(ok.kU > b), 3))
    for nm, mm in [("psi<2", ok.psi < 2), ("psi>=2", ok.psi >= 2)]:
        print(f"  {nm} n={mm.sum()} Lss", q(ok.Lss[mm]), "LE", q(ok.LE[mm]))
    if cfg == "QD":
        for lo, hi in [(-9, 0), (0, 0.27), (0.27, 0.5), (0.5, 1)]:
            mm = (ok.eps >= lo) & (ok.eps < hi); print(f"  eps in [{lo},{hi}) n={mm.sum()} Lss", q(ok.Lss[mm]), "LE", q(ok.LE[mm]))
    npz = g[g.status == "no price"]
    print("  no price: median psi", round(npz.psi.median(), 2), "M", round(npz.M.median(), 1), "| ok: psi", round(ok.psi.median(), 2), "M", round(ok.M.median(), 1))
    print("  no-price share by psi bin:", g.assign(b=pd.cut(g.psi, [1, 1.5, 2, 3, 5, 20])).groupby("b", observed=True).status.apply(lambda s: round(np.mean(s == "no price"), 2)).to_dict())
    # comparison with midpoint run on common draws
    j = ok.set_index("i").join(m, how="inner"); j = j[j.status_mid == "ok"]
    print(f"  common draws with midpoint run: {len(j)}; median L/E quarter {np.median(j.kF/j.kU):.2f} vs midpoint {np.median(j.kF_mid/j.kU_mid):.2f}; median L/ss quarter {np.median(j.kF/j.kstar):.2f} vs midpoint {np.median(j.kF_mid/j.kstar):.2f}")
from scipy.stats import spearmanr
ok = o[(o.cfg == "QD") & (o.status == "ok")].copy(); ok["Lss"] = ok.kF / ok.kstar
for c in ["psi", "eps", "s", "sigma_A", "Theta", "r_f", "alpha", "rho", "Eshare"]:
    print("spearman Lss vs", c, round(spearmanr(ok.Lss, ok[c])[0], 3))
