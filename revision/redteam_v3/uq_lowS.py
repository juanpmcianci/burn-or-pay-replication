"""Red team v3: the D7 design with the fee speed drawn on logU[0.3, 1] instead of logU[1, 5], the range that the
within-chain fee regression cannot exclude once demand shocks are allowed for (fee_sim.txt). Same 400 draws,
same sigma_A draws as uq2 (seed 1000+i); s re-drawn with seed 5000+i. Combined with QD it gives logU[0.3, 5].
Run from revision/redteam_v3."""
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
from multiprocessing import Pool
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent / "uq"))
import os; os.chdir(R)
from uq1 import params
from centred import gap, solve_c, with_, fee

def one(a):
    i, row = a
    p = params(row, 0.0); F = p.F_max / 2
    g = np.random.default_rng(1000 + int(i)); g.uniform(); sig = float(g.uniform(0.15, 0.50))
    s = float(np.exp(np.random.default_rng(5000 + int(i)).uniform(np.log(0.3), np.log(1.0))))
    p = with_(p, F_max=4 * F, F_min=0.0, s=s, x0=-s * np.log(3.0), sigma_A=sig)
    out = dict(i=i, s=s, sigma_A=sig)
    try:
        for kind in "FU":
            g0 = gap(p, 0.0, kind, kappa=0.0); g2 = gap(p, 0.0, kind, kappa=0.2)
            out[f"k{kind}"] = -g0 / ((g2 - g0) / 0.2)
        lw = solve_c(p, p); h = 1e-3
        q = (lw(h)[0] - lw(-h)[0]) / (2 * h)
        dlF = (np.log(fee(np.array([h]), p)) - np.log(fee(np.array([-h]), p)))[0] / (2 * h)
        out["eps"] = 1 + dlF / q; out["status"] = "ok"
    except Exception as e:
        out["status"] = "no price" if "no price" in str(e) else f"error: {str(e)[:40]}"
    return out

if __name__ == "__main__":
    o = pd.read_parquet(R / "output/uq2_draws.parquet"); qd = o[o.cfg == "QD"].copy()
    d0 = pd.read_parquet(R.parent / "uq/output/uq0_draws.parquet")
    args = [(int(i), d0.loc[int(i)]) for i in qd.i]
    with Pool(10) as pool:
        r = pd.DataFrame(pool.map(one, args, chunksize=2))
    r = r.merge(qd[["i", "kstar", "r", "psi"]], on="i")
    r.to_csv(Path(__file__).parent / "uq_lowS.csv", index=False)
    ok = r[r.status == "ok"].copy(); L = ok.kF / ok.kstar
    qq = lambda x: np.round(x.quantile([.05, .25, .5, .75, .95]).values, 2)
    print("status:", r.status.value_counts().to_dict())
    print("logU[0.3,1]: kL/k*", qq(L), "within10", round(np.mean(abs(L - 1) < .1), 3), "within20", round(np.mean(abs(L - 1) < .2), 3), "| kL/kE", qq(ok.kF / ok.kU), "| eps", qq(ok.eps))
    # combine with QD into logU[0.3,5]: weight low part by ln(1/0.3)/ln(5/0.3), high part by ln5/ln(5/0.3), per draw
    wl, wh = np.log(1 / 0.3) / np.log(5 / 0.3), np.log(5) / np.log(5 / 0.3)
    qd["ok"] = qd.status == "ok"; ok_h = qd[qd.ok]
    in_h = np.sum(abs(ok_h.kF / ok_h.kstar - 1) < .1); in_l = np.sum(abs(L - 1) < .1)
    pr = wl * len(ok) / len(r) + wh * len(ok_h) / len(qd)
    print(f"logU[0.3,5] mixture: priced share {pr:.2f}; within 10% among priced {(wl*in_l/len(r) + wh*in_h/len(qd))/pr:.3f}; among all draws {wl*in_l/len(r) + wh*in_h/len(qd):.3f}")
    allL = np.r_[L.values, (ok_h.kF / ok_h.kstar).values]; w = np.r_[np.full(len(L), wl / len(r)), np.full(len(ok_h), wh / len(qd))]
    i = np.argsort(allL); c = np.cumsum(w[i]) / w.sum(); print("   median kL/k* among priced", round(allL[i][np.searchsorted(c, .5)], 3), " p90", round(allL[i][np.searchsorted(c, .9)], 3))
