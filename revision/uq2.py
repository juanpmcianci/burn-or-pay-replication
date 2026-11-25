"""UQ level 1, rerun (UQ_PLAN amendment D7): reference state at a quarter of F_max, as in Table 5.
Config Q75: s = 0.75, sigma_A = 0.35 (Table 5's fee path, for comparison with the midpoint run uq1c).
Config QD : s ~ logU[1, 5], sigma_A ~ U[0.15, 0.50] (fee speeds whose dollar-fee elasticity lies in the range
            the fee data allow; see feedata/elasticity.py).
Fee path: F_max = 4F, F_min = 0, x0 = -s ln 3, so F(0) = F at each draw's reference state z = 0.
Per draw: burn-share and issuance hurdles (finite differences, centred feedback), closed form kstar, the
dollar-fee elasticity eps at z = 0, and the Proposition 2 formula hurdles as a check."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "uq"))
from uq1 import params            # noqa: E402
from centred import gap, solve_c, with_, fee   # noqa: E402
import prop2                      # noqa: E402

def setup(row, cfg, i):
    p = params(row, 0.0)
    F = p.F_max / 2
    if cfg == "Q75":
        s, sig = 0.75, 0.35
    else:
        g = np.random.default_rng(1000 + int(i))
        s, sig = float(np.exp(g.uniform(np.log(1.0), np.log(5.0)))), float(g.uniform(0.15, 0.50))
    return with_(p, F_max=4 * F, F_min=0.0, s=s, x0=-s * np.log(3.0), sigma_A=sig), s, sig

def one(args):
    i, row, cfg = args
    p, s, sig = setup(row, cfg, i)
    out = dict(i=i, cfg=cfg, s=s, sigma_A=sig)
    try:
        for kind in "FU":
            g0 = gap(p, 0.0, kind, kappa=0.0); g2 = gap(p, 0.0, kind, kappa=0.2)
            S = (g2 - g0) / 0.2
            out[f"G0{kind}"], out[f"S{kind}"] = g0, S
            out[f"k{kind}"] = -g0 / S if S > 0 else np.nan
        lw = solve_c(p, p); h = 1e-3
        q = (lw(h)[0] - lw(-h)[0]) / (2 * h)
        dlF = (np.log(fee(np.array([h]), p)) - np.log(fee(np.array([-h]), p)))[0] / (2 * h)
        out["q"], out["eps"] = q, 1 + dlF / q
        # Proposition 2 formula at z = 0
        P = prop2.pieces(p); z = P["z"]; i0 = int(np.argmin(np.abs(z))); Fr = fee(np.array([z[i0]]), p)[0]
        Me = 1 / (p.alpha * P["D"][[0, -1]])
        for kind, dT in [("U", P["dth"]), ("F", P["dth"] * P["F"] / Fr)]:
            vh, _ = prop2.solve_lin(P, dT * P["h"], dT[[0, -1]] * Me * P["kcl"][[0, -1]], p.sigma_A)
            vq, _ = prop2.solve_lin(P, dT * P["q"], dT[[0, -1]] * Me, p.sigma_A)
            out[f"k{kind}_p2"] = vh[i0] / vq[i0]
        out["qmin"] = P["q"].min()
        out["status"] = "ok"
    except Exception as e:
        out["status"] = "no price" if "no price" in str(e) else f"error: {str(e)[:50]}"
    return out

if __name__ == "__main__":
    d = pd.read_parquet(Path(__file__).resolve().parents[1] / "uq/output/uq0_draws.parquet")
    sub = d.sample(400, random_state=23)          # same 400 draws as uq1c
    args = [(i, r, cfg) for cfg in ["QD", "Q75"] for i, r in sub.iterrows()]
    with Pool(10) as pool:
        res = pool.map(one, args, chunksize=2)
    r = pd.DataFrame(res)
    o = r.merge(sub, left_on="i", right_index=True)
    o.to_parquet("output/uq2_draws.parquet")
    print("done", o.groupby("cfg").status.value_counts().to_dict())
