"""Review round 4, items 16-17. UQ rerun with (a) the price condition of centred2 (Gordon limit at the top, R > 0 at
the bottom) instead of 'Gordon denominator positive everywhere', (b) a domain wide enough that the fee path is
saturated at both ends (z in [min(-7, x0 - 10 s), max(7, x0 + 10 s)]), and (c) 1600 draws: the 400 of uq2 plus 1200
new ones from the same prior pool. Same two configurations as uq2 (QD, Q75)."""
import sys, numpy as np, pandas as pd, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "uq"))
from uq2 import setup                       # noqa: E402
from centred import fee, with_              # noqa: E402
from centred2 import gap2, solve_c2, NoPrice  # noqa: E402

def dom(p):
    return dict(x_lo=min(-7.0, p.x0 - 10 * p.s), x_hi=max(7.0, p.x0 + 10 * p.s), n=1201)

def one(args):
    i, row, cfg, batch = args
    p, s, sig = setup(row, cfg, i)
    out = dict(i=i, cfg=cfg, batch=batch, s=s, sigma_A=sig)
    t = time.time()
    try:
        kw = dom(p)
        for kind in "FU":
            g0 = gap2(p, 0.0, kind, 0.0, **kw); g2 = gap2(p, 0.0, kind, 0.2, **kw)
            S = (g2 - g0) / 0.2
            out[f"G0{kind}"], out[f"S{kind}"] = g0, S
            out[f"k{kind}"] = -g0 / S if S > 0 else np.nan
        lw = solve_c2(p, p, **kw); h = 1e-3
        q = (lw(h)[0] - lw(-h)[0]) / (2 * h)
        dlF = (np.log(fee(np.array([h]), p)) - np.log(fee(np.array([-h]), p)))[0] / (2 * h)
        out["q"], out["eps"] = q, 1 + dlF / q
        out["status"] = "ok"
    except NoPrice as e:
        out["status"] = "no price"; out["why"] = str(e)[:40]
    except Exception as e:
        out["status"] = f"error: {str(e)[:50]}"
    out["sec"] = time.time() - t
    return out

def draws():
    pool = pd.read_parquet(Path(__file__).resolve().parents[1] / "uq/output/uq0_draws.parquet")
    a = pool.sample(400, random_state=23)
    b = pool.drop(a.index).sample(1200, random_state=24)
    return pd.concat([a.assign(batch=0), b.assign(batch=1)])

if __name__ == "__main__":
    sub = draws()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(sub)
    sub = sub.head(n)
    args = [(i, r, cfg, int(r.batch)) for cfg in ["QD", "Q75"] for i, r in sub.iterrows()]
    with Pool(10) as pool:
        res = pool.map(one, args, chunksize=2)
    o = pd.DataFrame(res).merge(sub.drop(columns="batch"), left_on="i", right_index=True)
    o.to_parquet(f"output/uq3_draws{'' if n == 1600 else '_test'}.parquet")
    print("done", o.groupby("cfg").status.value_counts().to_dict(), "mean sec", o.sec.mean())
