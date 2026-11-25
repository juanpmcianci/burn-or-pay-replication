"""Red team v3: robustness of the fee-elasticity regressions (feedata/elasticity.py) and the data moments the
model's mapping needs. Reads ../feedata/fees_mcap_daily.csv; writes nothing to the paper's outputs."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
d = pd.read_csv("../feedata/fees_mcap_daily.csv", parse_dates=["date"])
d = d[(d.fee > 0) & (d.mcap > 0)]
d["year"] = d.date.dt.year; d["q"] = d.date.dt.to_period("Q")
span = d.groupby("chain").date.agg(["min", "max", "size"])
print("chains:", d.chain.nunique()); print(span.sort_values("min").to_string())

def annual(d):
    y = d.groupby(["chain", "year"]).agg(fee=("fee", "sum"), mcap=("mcap", "mean"), n=("fee", "size")).reset_index()
    y = y[y.n >= 300].copy(); y["lf"] = np.log(y.fee * 365 / y.n); y["lm"] = np.log(y.mcap)
    y = y.sort_values(["chain", "year"]); y["dlf"] = y.groupby("chain").lf.diff(); y["dlm"] = y.groupby("chain").lm.diff()
    y["gap"] = y.groupby("chain").year.diff()
    return y

def cl(f, data, g):
    r = smf.ols(f, data).fit(cov_type="cluster", cov_kwds={"groups": data[g].astype("category").cat.codes})
    return r

def wild(f, data, var, g, B=1999, seed=0):
    """Wild cluster bootstrap (Rademacher, null imposed) p-value-free percentile-t CI for coefficient `var`."""
    rng = np.random.default_rng(seed)
    r = cl(f, data, g); b, se = r.params[var], r.bse[var]
    codes = data[g].astype("category").cat.codes.values; G = codes.max() + 1
    X = r.model.exog; yhat = r.fittedvalues.values; e = r.resid.values
    ts = []
    for _ in range(B):
        w = rng.choice([-1, 1], G)[codes]
        ystar = yhat + e * w
        rb = smf.ols(f, data.assign(**{r.model.endog_names: ystar})).fit(cov_type="cluster", cov_kwds={"groups": codes})
        ts.append((rb.params[var] - b) / rb.bse[var])
    lo, hi = np.quantile(ts, [0.975, 0.025])
    return b, se, b - lo * se, b - hi * se, G

y = annual(d); yy = y.dropna(subset=["dlf"])
print("\nannual obs", len(y), "chains", y.chain.nunique(), "| within-annual obs", len(yy), "chains", yy.chain.nunique(), "non-consecutive gaps", int((yy.gap > 1).sum()))
print("chain-year counts by year:", y.groupby("year").size().to_dict())
r = cl("dlf ~ dlm + C(year)", yy, "chain")
G = yy.chain.nunique()
from scipy.stats import t
print(f"within annual, year FE: beta {r.params.dlm:.3f} se {r.bse.dlm:.3f}; normal 95% [{r.params.dlm-1.96*r.bse.dlm:.2f}, {r.params.dlm+1.96*r.bse.dlm:.2f}]; t(G-1={G-1}) [{r.params.dlm-t.ppf(.975,G-1)*r.bse.dlm:.2f}, {r.params.dlm+t.ppf(.975,G-1)*r.bse.dlm:.2f}]")
b, se, lo, hi, G = wild("dlf ~ dlm + C(year)", yy, "dlm", "chain", B=999)
print(f"   wild cluster bootstrap-t 95% CI: [{lo:.2f}, {hi:.2f}] (G={G})")
r2 = cl("dlf ~ dlm", yy, "chain"); print(f"within annual, NO year FE: beta {r2.params.dlm:.3f} se {r2.bse.dlm:.3f}")
r3 = cl("dlf ~ dlm + C(year)", yy[yy.year < 2026], "chain"); print(f"within annual, year FE, drop 2026: beta {r3.params.dlm:.3f} se {r3.bse.dlm:.3f} n {int(r3.nobs)}")
r4 = cl("dlf ~ dlm + C(year)", yy[~((yy.chain == 'ethereum') & (yy.year <= 2022))], "chain"); print(f"within annual, year FE, drop Ethereum PoW years: beta {r4.params.dlm:.3f} se {r4.bse.dlm:.3f}")
r5 = cl("dlf ~ dlm + C(year)", yy[yy.gap == 1], "chain"); print(f"within annual, consecutive years only: beta {r5.params.dlm:.3f} se {r5.bse.dlm:.3f}")
loo = [cl("dlf ~ dlm + C(year)", yy[yy.chain != c], "chain").params.dlm for c in yy.chain.unique()]
print(f"   leave-one-chain-out range: {min(loo):.2f} to {max(loo):.2f}")
# cross-section
rc = cl("lf ~ lm + C(year)", y, "chain"); rc0 = cl("lf ~ lm", y, "chain")
print(f"cross-section pooled, year FE: {rc.params.lm:.3f} ({rc.bse.lm:.3f}); no year FE {rc0.params.lm:.3f} ({rc0.bse.lm:.3f})")
rc2 = cl("lf ~ lm + C(year) + C(chain)", y, "chain"); print(f"levels with chain + year FE (within): {rc2.params.lm:.3f} ({rc2.bse.lm:.3f})")
# fee yield moments
y["ly"] = y.lf - y.lm
print("\nmedian fee yield by year:", y.groupby("year").ly.median().apply(np.exp).round(5).to_dict())
yy = y.dropna(subset=["dlf"]).copy(); yy["dly"] = yy.dlf - yy.dlm
res = lambda v: smf.ols(f"{v} ~ C(year)", yy).fit().resid
print("within-annual sd after year FE: dlog mcap %.2f, dlog fees %.2f, dlog yield %.2f; raw sd dlog mcap %.2f, dlog yield %.2f; mean dlog yield %.2f"
      % (res("dlm").std(), res("dlf").std(), res("dly").std(), yy.dlm.std(), yy.dly.std(), yy.dly.mean()))
print("corr(dlog yield, dlog mcap) after year FE: %.2f" % np.corrcoef(res("dly"), res("dlm"))[0, 1])
print("share of within-annual dlog mcap variance absorbed by year FE: %.2f" % (1 - res("dlm").var() / yy.dlm.var()))
# quarterly
q = d.groupby(["chain", "q"]).agg(fee=("fee", "sum"), mcap=("mcap", "mean"), n=("fee", "size")).reset_index()
q = q[q.n >= 85]; q["lf"] = np.log(q.fee / q.n); q["lm"] = np.log(q.mcap)
q = q.sort_values(["chain", "q"]); q["dlf"] = q.groupby("chain").lf.diff(); q["dlm"] = q.groupby("chain").lm.diff()
qq = q.dropna(subset=["dlf"]).copy(); qq["qs"] = qq.q.astype(str)
rq = cl("dlf ~ dlm + C(qs)", qq, "chain"); rq0 = cl("dlf ~ dlm", qq, "chain")
print(f"\nquarterly within, quarter FE: {rq.params.dlm:.3f} ({rq.bse.dlm:.3f}); no FE {rq0.params.dlm:.3f} ({rq0.bse.dlm:.3f}); G={qq.chain.nunique()}")
b, se, lo, hi, G = wild("dlf ~ dlm + C(qs)", qq, "dlm", "chain", B=499)
print(f"   wild cluster bootstrap-t 95% CI: [{lo:.2f}, {hi:.2f}]")
# lagged response: fees respond to mcap with a lag? distributed lag on quarterly changes
qq["dlm1"] = qq.groupby("chain").dlm.shift(1); qq["dlm2"] = qq.groupby("chain").dlm.shift(2)
rl = cl("dlf ~ dlm + dlm1 + dlm2 + C(qs)", qq.dropna(subset=["dlm2"]), "chain")
print(f"   distributed lag: contemporaneous {rl.params.dlm:.2f}, lag1 {rl.params.dlm1:.2f}, lag2 {rl.params.dlm2:.2f}, sum {rl.params.dlm+rl.params.dlm1+rl.params.dlm2:.2f}")
