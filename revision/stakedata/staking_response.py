"""Review round 4, items 20 and 25.
(a) Staking response: within-chain regressions of the staking ratio on the issuance rate paid to stakers E (quarterly
means), chain fixed effects, SE clustered by chain; and the cross-section of current Theta on current rho = r Theta.
(b) Growth on stake: chain-level mean annual log change in market cap (after year effects) on mean Theta."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
d = pd.read_csv("staking_panel.csv", parse_dates=["date"])
d = d[~((d.chain == "sui") & (d.date.dt.year == 2023))]
d["q"] = d.date.dt.to_period("Q")
qd = d.dropna(subset=["theta", "E"]).groupby(["chain", "q"]).agg(theta=("theta", "mean"), E=("E", "mean"), r=("r", "mean")).reset_index()
cnt = qd.groupby("chain").size(); panel = qd[qd.chain.isin(cnt[cnt >= 4].index)]
out = []
for lab, sub in [("all with >=4 quarters", panel), ("excl. fantom (Sonic migration)", panel[panel.chain != "fantom"]),
                 ("excl. fantom, sui (launch)", panel[~panel.chain.isin(["fantom", "sui"])])]:
    m = smf.ols("theta ~ E + C(chain)", sub).fit(cov_type="cluster", cov_kwds={"groups": sub.chain})
    s2 = sub.sort_values(["chain", "q"]).copy(); s2["dth"] = s2.groupby("chain").theta.diff(); s2["dE"] = s2.groupby("chain").E.diff(); s2 = s2.dropna(subset=["dth", "dE"])
    m2 = smf.ols("dth ~ dE", s2).fit(cov_type="cluster", cov_kwds={"groups": s2.chain})
    out.append(dict(spec=lab, chains=sub.chain.nunique(), n=len(sub), levels=m.params.E, se=m.bse.E, diffs=m2.params.dE, se_d=m2.bse.dE))
    print(out[-1])
cur = d.sort_values("date").groupby("chain").tail(1).dropna(subset=["theta", "r"]).copy(); cur["rho"] = cur.r * cur.theta
mc = smf.ols("theta ~ rho", cur).fit(cov_type="HC1")
print("cross-section current: n", len(cur), "slope dTheta/drho", round(mc.params.rho, 2), "se", round(mc.bse.rho, 2))
print("current theta quantiles", np.round(cur.theta.quantile([0, .25, .5, .75, 1]).values, 2), "r quantiles", np.round(cur.r.quantile([0, .25, .5, .75, 1]).values, 3), "rho q", np.round(cur.rho.quantile([.25, .5, .75]).values, 3))
pd.DataFrame(out).to_csv("staking_response.csv", index=False)
# (b) growth on stake
f = pd.read_csv("../feedata/fees_mcap_daily.csv", parse_dates=["date"])
f["y"] = f.date.dt.year; a = f.groupby(["chain", "y"]).agg(m=("mcap", "mean"), n=("mcap", "size")).reset_index(); a = a[a.n >= 300]
a = a.sort_values(["chain", "y"]); a["g"] = a.groupby("chain").m.transform(lambda x: np.log(x).diff()); a = a.dropna()
a["g_adj"] = a.g - a.groupby("y").g.transform("mean")
th = d.groupby("chain").theta.mean().rename("theta_mean")
c = a.groupby("chain").g_adj.mean().to_frame().join(th, how="inner")
mg = smf.ols("g_adj ~ theta_mean", c).fit(cov_type="HC1")
print("growth on mean theta: chains", len(c), "slope", round(mg.params.theta_mean, 3), "se", round(mg.bse.theta_mean, 3),
      "95%", np.round(mg.conf_int().loc["theta_mean"].values, 2), "| sd theta", round(c.theta_mean.std(), 3), "sd growth", round(c.g_adj.std(), 3))
