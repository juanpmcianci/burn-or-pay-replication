"""Elasticity of dollar fee revenue with respect to market capitalisation, PoS L1s.
(1) cross-section by calendar year: log annual fees on log mean market cap;
(2) within chain, annual (non-overlapping, 365-day) log changes, pooled, with year effects;
(3) within chain, quarterly log changes (shorter horizon).
Also: fee yield F = fees/mcap distribution."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
d = pd.read_csv("fees_mcap_daily.csv", parse_dates=["date"])
d = d[(d.fee > 0) & (d.mcap > 0)]
d["year"] = d.date.dt.year; d["q"] = d.date.dt.to_period("Q")
out = []
# (1) cross-section by year, full years with >= 300 days
y = d.groupby(["chain", "year"]).agg(fee=("fee", "sum"), mcap=("mcap", "mean"), n=("fee", "size")).reset_index()
y = y[y.n >= 300]; y["lf"] = np.log(y.fee * 365 / y.n); y["lm"] = np.log(y.mcap); y["yield"] = np.exp(y.lf) / y.mcap
print("Cross-section by year: log annual fees on log market cap")
for yr, g in y.groupby("year"):
    if len(g) < 6: continue
    r = smf.ols("lf ~ lm", g).fit(cov_type="HC1")
    print(f"  {yr}: n={len(g):2d}  beta={r.params.lm:.2f} (se {r.bse.lm:.2f})  median fee yield={g['yield'].median():.4f}")
    out.append(dict(kind="xsec", year=yr, n=len(g), beta=r.params.lm, se=r.bse.lm))
r = smf.ols("lf ~ lm + C(year)", y).fit(cov_type="cluster", cov_kwds={"groups": y.chain.astype("category").cat.codes})
print(f"  pooled with year effects: n={len(y)} beta={r.params.lm:.2f} (se {r.bse.lm:.2f}, clustered by chain)")
out.append(dict(kind="xsec_pooled", n=len(y), beta=r.params.lm, se=r.bse.lm))
# (2) within chain annual changes
y = y.sort_values(["chain", "year"]); y["dlf"] = y.groupby("chain").lf.diff(); y["dlm"] = y.groupby("chain").lm.diff()
yy = y.dropna(subset=["dlf"])
r = smf.ols("dlf ~ dlm + C(year)", yy).fit(cov_type="cluster", cov_kwds={"groups": yy.chain.astype("category").cat.codes})
print(f"Within chain, annual log changes: n={len(yy)} chains={yy.chain.nunique()} beta={r.params.dlm:.2f} (se {r.bse.dlm:.2f})")
out.append(dict(kind="within_annual", n=len(yy), beta=r.params.dlm, se=r.bse.dlm))
r2 = smf.ols("dlf ~ dlm", yy).fit(cov_type="cluster", cov_kwds={"groups": yy.chain.astype("category").cat.codes})
print(f"   without year effects: beta={r2.params.dlm:.2f} (se {r2.bse.dlm:.2f})")
# (3) quarterly
q = d.groupby(["chain", "q"]).agg(fee=("fee", "sum"), mcap=("mcap", "mean"), n=("fee", "size")).reset_index()
q = q[q.n >= 85]; q["lf"] = np.log(q.fee / q.n); q["lm"] = np.log(q.mcap)
q = q.sort_values(["chain", "q"]); q["dlf"] = q.groupby("chain").lf.diff(); q["dlm"] = q.groupby("chain").lm.diff()
qq = q.dropna(subset=["dlf"]); qq["qs"] = qq.q.astype(str)
r = smf.ols("dlf ~ dlm + C(qs)", qq).fit(cov_type="cluster", cov_kwds={"groups": qq.chain.astype("category").cat.codes})
print(f"Within chain, quarterly log changes: n={len(qq)} beta={r.params.dlm:.2f} (se {r.bse.dlm:.2f})")
out.append(dict(kind="within_quarterly", n=len(qq), beta=r.params.dlm, se=r.bse.dlm))
# per-chain time-series long-run: regress log quarterly fees on log mcap, chains with >= 12 quarters
print("Per chain, quarterly levels (>= 12 quarters):")
pc = []
for c, g in q.groupby("chain"):
    if len(g) < 12: continue
    rr = smf.ols("lf ~ lm", g).fit(); pc.append(dict(chain=c, nq=len(g), beta=rr.params.lm))
pc = pd.DataFrame(pc).sort_values("beta"); print(pc.round(2).to_string(index=False))
print("  median per-chain beta:", round(pc.beta.median(), 2), " IQR:", pc.beta.quantile([.25, .75]).round(2).tolist())
pd.DataFrame(out).to_csv("elasticity.csv", index=False); pc.to_csv("elasticity_perchain.csv", index=False)
