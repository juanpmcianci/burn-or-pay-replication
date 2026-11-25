"""Outputs of UQ_PLAN amendment D5, from output/uq1f_draws.parquet."""
import numpy as np, pandas as pd
o = pd.read_parquet("output/uq1f_draws.parquet")
n = len(o)
print("status:", o.status.value_counts().to_dict())
priced = o[o.status.isin(["root", "no root"])]
ok = o[o.status == "root"].copy()
q = lambda s: np.round(s.quantile([.05, .5, .95]).values, 2)
print(f"priced at kappa=0: {len(priced)} of {n}; roots {len(ok)} ({len(ok)/len(priced):.3f}); no root {int((o.status=='no root').sum())}")
nr = o[o.status == "no root"]
print(f"  no-root draws with kmax < r: {int((nr.kmax < nr.r).sum())}; kmax capped at 1.5 among priced: {int(priced.capped.sum())}")
ok["rF"] = ok.kF / ok.r
print("kF/r 5/50/95:", q(ok.rF), "| kU/r on same draws:", q((ok.kU / ok.r).dropna()), f"(kU found in {ok.kU.notna().sum()})")
print("kF/kstar(steady) 5/50/95:", q(ok.kF / ok.kstar), "| kU/kstar:", q((ok.kU / ok.kstar).dropna()))
both = ok.dropna(subset=["kU"])
print(f"kF > kU in {np.mean(both.kF > both.kU):.3f} of {len(both)}; median kF-kU {np.median(both.kF-both.kU):+.3f}")
edge = (ok.kmax - ok.kF) / ok.kmax
print("edge distance 5/50/95:", q(edge), f"| share within 5% of edge: {np.mean(edge < 0.05):.3f}; within 10%: {np.mean(edge<0.10):.3f}")
b = ok.r + ok.Gamma / (1 - ok.Theta)
m = ok.kF - b
print(f"bound kF > r + Gamma/(1-Theta): {np.mean(m > 0):.3f} of roots; min margin {m.min():+.3f}; kF > r: {np.mean(ok.kF > ok.r):.3f}")
print(f"kF > steady kstar: {np.mean(ok.kF > ok.kstar):.3f}")
for nm, mm in [("psi<2", ok.psi < 2), ("psi>=2", ok.psi >= 2)]:
    print(f"{nm}: n={mm.sum()} kF/r median {ok.rF[mm].median():.2f} 5/95 {q(ok.rF[mm])}")
# no-root share by psi
for nm, mm in [("psi<2", priced.psi < 2), ("psi>=2", priced.psi >= 2)]:
    print(f"{nm}: no-root share {np.mean(priced.status[mm]=='no root'):.3f} (n={mm.sum()})")
print("issuance root share among priced:", np.mean(priced.kU.notna()).round(3))
