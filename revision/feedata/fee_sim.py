"""Red team v3: does the within-chain fee regression recover the lifecycle dollar-fee elasticity in the model?
Simulate panels from the paper's centred solver (baseline economy, omega = 0), with dollar fees = F(z) W(z) S and
market cap = W(z) S, a common demand shock (absorbed by year effects) and an idiosyncratic one sized so that the
within-chain sd of annual log mcap changes after year effects matches the data (0.60, fee_audit.txt).
Run the paper's regression: annual log changes, year effects, SE clustered by chain. Compare with eps at reference.
Run from revision/redteam_v3 (imports ../centred.py)."""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/
from centred import solve_c, calibrated, fee
from feasible import x_at_theta

def panel(lw, p, x0, rng, nchain=18, nyear=5, sd_target=0.60, steps=52):
    sig = p.sigma_A; dt = 1 / steps
    q_ref = (lw(x0 + 1e-3)[0] - lw(x0 - 1e-3)[0]) / 2e-3
    sd_idio = np.sqrt(max(sd_target**2 - (q_ref * sig) ** 2, 0.01))
    z = x0 + rng.uniform(-1, 1, nchain)
    rows = []
    lS = rng.normal(0, 1, nchain)
    for yr in range(nyear):
        fees = np.zeros(nchain); caps = np.zeros(nchain)
        common = rng.normal(0, 0.6 * np.sqrt(dt), steps)
        for k in range(steps):
            z = z + (p.mu0 - sig**2 / 2) * dt + sig * np.sqrt(dt) * rng.normal(size=nchain)
            z = np.clip(z, -6.9, 6.9)
            lS = lS + common[k] + sd_idio * np.sqrt(dt) * rng.normal(size=nchain)
            cap = np.exp(lw(z) + lS); fees += fee(z, p) * cap * dt; caps += cap * dt
        for c in range(nchain):
            rows.append(dict(chain=c, year=yr, lf=np.log(fees[c]), lm=np.log(caps[c])))
    d = pd.DataFrame(rows).sort_values(["chain", "year"])
    d["dlf"] = d.groupby("chain").lf.diff(); d["dlm"] = d.groupby("chain").lm.diff(); d = d.dropna()
    r = smf.ols("dlf ~ dlm + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": d.chain})
    return r.params.dlm, r.bse.dlm, q_ref, sd_idio

if __name__ == "__main__":
    out = []
    for name, psi, rf, s in [("A", 1.5, 0.20, 0.25), ("A", 1.5, 0.20, 0.75), ("B", 11.2, 0.20, 0.25), ("B", 11.2, 0.20, 0.4), ("B", 11.2, 0.20, 0.75)]:
        p = calibrated(beta=psi, kappa=0.0, r_f=rf, F_max=0.12, F_min=0.0, s=s)
        x0 = x_at_theta(p); lw = solve_c(p, p)
        h = 1e-3; q = (lw(x0 + h)[0] - lw(x0 - h)[0]) / (2 * h)
        dlF = (np.log(fee(np.array([x0 + h]), p)) - np.log(fee(np.array([x0 - h]), p)))[0] / (2 * h)
        eps = 1 + dlF / q
        rng = np.random.default_rng(7)
        R = np.array([panel(lw, p, x0, rng)[:2] for _ in range(300)])
        b, se = R[:, 0], R[:, 1]
        out.append(dict(case=name, psi=psi, s=s, eps_lifecycle=eps, q_ref=q, median_beta_hat=np.median(b),
                        p05=np.quantile(b, .05), p95=np.quantile(b, .95), share_lower95_ge_027=np.mean(b - 1.96 * se >= 0.27),
                        share_beta_ge_094=np.mean(b >= 0.94)))
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv("fee_sim.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))
