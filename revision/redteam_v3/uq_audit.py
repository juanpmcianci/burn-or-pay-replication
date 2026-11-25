"""Red team v3: audit of the D7 UQ run (output/uq2_draws.parquet), no new solves.
(1) headline share within 10% under alternative denominators (selection);
(2) inverse-probability reweighting of priced draws by psi bin and s bin;
(3) what the headline would be under alternative priors on s (reweighting within [1,5]) and by eps bin;
(4) no-price rate against s (does the s prior interact with selection?)."""
import numpy as np, pandas as pd
o = pd.read_parquet("../output/uq2_draws.parquet")
g = o[o.cfg == "QD"].copy()
g["ok"] = g.status == "ok"
ok = g[g.ok].copy(); ok["Lss"] = ok.kF / ok.kstar; ok["w10"] = (ok.Lss - 1).abs() < 0.1; ok["w20"] = (ok.Lss - 1).abs() < 0.2
P = lambda x: f"{100*x:.1f}%"
print("QD draws:", len(g), "priced:", ok.shape[0])
print("(1) within 10%: of priced", P(ok.w10.mean()), "| of all 400, unpriced counted as not within", P(ok.w10.sum() / len(g)))
print("    within 20%: of priced", P(ok.w20.mean()), "| of all 400", P(ok.w20.sum() / len(g)))
print("    share of priced with Lss>1.1:", P((ok.Lss > 1.1).mean()), " Lss<0.9:", P((ok.Lss < 0.9).mean()))
# (2) IPW by psi bin
bins = [1, 1.5, 2, 3, 5, 20]
g["pb"] = pd.cut(g.psi, bins); ok["pb"] = pd.cut(ok.psi, bins)
pr = g.groupby("pb", observed=True).ok.mean()
ok["ipw"] = 1 / ok.pb.map(pr).astype(float)
print("(2) price rate by psi bin:", pr.round(2).to_dict())
by = ok.groupby("pb", observed=True).agg(n=("w10", "size"), w10=("w10", "mean"), medLss=("Lss", "median"), p95=("Lss", lambda x: x.quantile(.95)))
print(by.round(3).to_string())
wm = lambda x, w: np.sum(x * w) / np.sum(w)
def wq(x, w, q):
    i = np.argsort(x.values); c = np.cumsum(w.values[i]) / w.sum(); return x.values[i][np.searchsorted(c, q)]
print("    IPW(psi) within 10%:", P(wm(ok.w10, ok.ipw)), " within 20%:", P(wm(ok.w20, ok.ipw)), " median Lss:", round(wq(ok.Lss, ok.ipw, .5), 3), " p95:", round(wq(ok.Lss, ok.ipw, .95), 3))
# (4) no-price rate vs s
g["sb"] = pd.cut(g.s, [1, 1.5, 2.2, 3.3, 5])
print("(4) price rate by s bin:", g.groupby("sb", observed=True).ok.mean().round(3).to_dict())
print("    price rate by s bin, psi>=3 only:", g[g.psi >= 3].groupby("sb", observed=True).ok.mean().round(3).to_dict())
ok["sb"] = pd.cut(ok.s, [1, 1.5, 2.2, 3.3, 5])
print("    within-10% by s bin (priced):", ok.groupby("sb", observed=True).w10.mean().round(3).to_dict())
print("    median Lss by s bin:", ok.groupby("sb", observed=True).Lss.median().round(3).to_dict())
# (3) alternative priors on s, via importance weights relative to logU[1,5] (density ∝ 1/s)
for name, dens in [("logU[1,5] (paper)", lambda s: 1 / s), ("U[1,5]", lambda s: np.ones_like(s)),
                   ("logU[1,2.5] (eps~0.25-0.6)", lambda s: (s <= 2.5) / s), ("logU[1,1.6]", lambda s: (s <= 1.6) / s)]:
    w = dens(ok.s.values) * ok.s.values  # target / proposal (proposal ∝ 1/s)
    w = pd.Series(w, index=ok.index)
    print(f"(3) s prior {name:28s}: within 10% {P(wm(ok.w10, w))}, within 20% {P(wm(ok.w20, w))}, median Lss {wq(ok.Lss, w, .5):.3f}, p90 {wq(ok.Lss, w, .9):.3f}; with psi IPW: within10 {P(wm(ok.w10, w*ok.ipw))}")
# eps bins
ok["eb"] = pd.cut(ok.eps, [-9, 0.27, 0.4, 0.5, 0.6, 0.7, 1])
print("(3b) by eps bin:\n", ok.groupby("eb", observed=True).agg(n=("w10", "size"), w10=("w10", "mean"), w20=("w20", "mean"), med=("Lss", "median")).round(3).to_string())
# q<0 hypothesis of Prop 2
print("(5) priced draws with qmin<0:", int((ok.qmin < 0).sum()), "of", len(ok), "; min qmin", round(ok.qmin.min(), 3))
print("    max |kF_p2-kF|", round((ok.kF_p2 - ok.kF).abs().max(), 4), " relative:", round(((ok.kF_p2 - ok.kF).abs() / ok.kF).max(), 4))
print("(6) Lss>1.1 among psi>=2:", P((ok[ok.psi >= 2].Lss > 1.1).mean()), " n psi>=2 priced", (ok.psi >= 2).sum(), " of psi>=2 drawn", (g.psi >= 2).sum())
