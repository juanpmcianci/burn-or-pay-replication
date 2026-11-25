"""Retry uq3 draws that hit solver errors with a looser tolerance (1e-6), 2401 nodes and a larger mesh cap."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
import uq3
def one(args):
    i, row, cfg, batch = args
    orig = uq3.dom
    uq3.dom = lambda p: dict(x_lo=min(-7.0, p.x0 - 10 * p.s), x_hi=max(7.0, p.x0 + 10 * p.s), n=2401, tol=1e-6)
    out = uq3.one(args); out["retry"] = True
    return out
if __name__ == "__main__":
    o = pd.read_parquet("output/uq3_draws.parquet"); sub = uq3.draws()
    bad = o[o.status.str.startswith("error")]
    args = [(r.i, sub.loc[r.i], r.cfg, int(r.batch)) for _, r in bad.iterrows()]
    with Pool(10) as pool: res = pool.map(one, args, chunksize=1)
    rr = pd.DataFrame(res); print(rr.groupby("cfg").status.value_counts().to_dict())
    keep = o[~o.status.str.startswith("error")]
    new = rr.merge(sub.drop(columns="batch"), left_on="i", right_index=True)
    pd.concat([keep, new]).to_parquet("output/uq3_draws.parquet")
