"""Red team v3: does the D7 headline depend on the lower boundary of the domain?
For priced QD draws whose fee path has not reached 90% of F_max at z = -7 (slow fades, s > 2.1), recompute the
burn-share and issuance hurdles with the lower boundary at -12 (fee yield there >= 0.86 F_max even at s = 5).
Same draws, same s and sigma_A (uq2.setup). Run from revision/redteam_v3."""
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
from multiprocessing import Pool
warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent / "uq"))
import os; os.chdir(R)
from uq2 import setup
from centred import gap

def one(a):
    i, row = a
    p, s, sig = setup(row, "QD", i)
    out = dict(i=i, s=s)
    for lo in [-7.0, -12.0]:
        try:
            k = {}
            for kind in "FU":
                g0 = gap(p, 0.0, kind, kappa=0.0, x_lo=lo); g2 = gap(p, 0.0, kind, kappa=0.2, x_lo=lo)
                k[kind] = -g0 / ((g2 - g0) / 0.2)
            out[f"kF_{-lo:.0f}"], out[f"kU_{-lo:.0f}"] = k["F"], k["U"]; out[f"st_{-lo:.0f}"] = "ok"
        except Exception as e:
            out[f"st_{-lo:.0f}"] = "no price" if "no price" in str(e) else str(e)[:40]
    return out

if __name__ == "__main__":
    o = pd.read_parquet(R / "output/uq2_draws.parquet"); g = o[(o.cfg == "QD") & (o.status == "ok")].copy()
    g["Flo"] = 1 / (1 + np.exp((-7 + g.s * np.log(3)) / g.s))
    sel = g[g.Flo < 0.9].sample(48, random_state=1)
    d0 = pd.read_parquet(R.parent / "uq/output/uq0_draws.parquet")
    args = [(int(i), d0.loc[int(i)]) for i in sel.i]
    with Pool(10) as pool:
        res = pd.DataFrame(pool.map(one, args))
    res = res.merge(sel[["i", "kstar", "psi"]], on="i")
    res.to_csv(Path(__file__).parent / "uq_domain.csv", index=False)
    print("status at -12:", res.st_12.value_counts().to_dict())
    ok = res[res.st_12 == "ok"]
    for lo in ["7", "12"]:
        L = ok[f"kF_{lo}"] / ok.kstar
        print(f"lower boundary -{lo}: n={len(ok)} median kL/k* {L.median():.3f}, within 10% {np.mean(abs(L-1)<0.1):.2f}, p90 {L.quantile(.9):.3f}")
    print("max |kF_12 - kF_7|:", (ok.kF_12 - ok.kF_7).abs().max().round(4), " median rel change:", ((ok.kF_12 - ok.kF_7) / ok.kF_7).median().round(4))
    np_ = res[res.st_12 != "ok"]; print("lose price at -12: n", len(np_), "median psi", np_.psi.median() if len(np_) else None,
          "; their kL/k* at -7: median", (np_.kF_7 / np_.kstar).median() if len(np_) else None)
    print("within 10% at -7 among all 48:", np.mean(abs(res.kF_7 / res.kstar - 1) < 0.1).round(2),
          "; counting price-loss at -12 as not within:", (np.sum(abs(ok.kF_12 / ok.kstar - 1) < 0.1) / len(res)).round(2))
