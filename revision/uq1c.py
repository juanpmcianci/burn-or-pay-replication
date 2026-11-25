"""UQ level 1 with base-path-centred feedback (UQ_PLAN amendment D6)."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "uq"))
from uq1 import params            # noqa: E402
from centred import gap           # noqa: E402

X = 0.0
def one(args):
    i, row, check = args
    p = params(row, 0.0)
    out = dict(i=i)
    try:
        for kind in "FU":
            g0 = gap(p, X, kind, kappa=0.0); g2 = gap(p, X, kind, kappa=0.2)
            s = (g2 - g0) / 0.2
            out[f"G0{kind}"], out[f"S{kind}"] = g0, s
            out[f"k{kind}"] = -g0 / s if s > 0 else np.nan
            if check:
                g4 = gap(p, X, kind, kappa=0.4)
                out[f"aff{kind}"] = (g4 - g0) / 0.4 / s
        out["status"] = "ok"
    except Exception as e:
        out["status"] = "no price" if "no price" in str(e) else f"error: {str(e)[:50]}"
    return out

if __name__ == "__main__":
    d = pd.read_parquet(Path(__file__).resolve().parents[1] / "uq/output/uq0_draws.parquet")
    sub = d.sample(400, random_state=23)
    args = [(i, r, j < 40) for j, (i, r) in enumerate(sub.iterrows())]
    with Pool(8) as pool:
        res = pool.map(one, args, chunksize=2)
    o = sub.join(pd.DataFrame(res).set_index("i"))
    o.to_parquet("output/uq1c_draws.parquet")
    print("done", o.status.value_counts().to_dict())
