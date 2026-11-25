"""Risk premium in the drift, on the 1,600 uq3 draws. Each draw's r_f is read as r0 + premium with r0 = 0.04 (the
bottom of the r_f prior): the agent discounts at r0 and productivity drift is lowered by the premium,
mu0 -> mu0 - (r_f - r0). Every steady state (D, kstar) is unchanged. Same configurations, domain and solver as uq3;
error draws are retried at tol 1e-6 with 2401 nodes, as in uq3_retry."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
import uq3
R0 = 0.04

def drift_row(row):
    r = row.copy(); r["mu0"] = row.mu0 - (row.r_f - R0); r["r_f"] = R0; return r

def one(args):
    i, row, cfg, batch, retry = args
    if retry:
        uq3.dom = lambda p: dict(x_lo=min(-7.0, p.x0 - 10 * p.s), x_hi=max(7.0, p.x0 + 10 * p.s), n=2401, tol=1e-6)
    out = uq3.one((i, drift_row(row), cfg, batch)); out["retry"] = retry
    return out

if __name__ == "__main__":
    sub = uq3.draws()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(sub)
    sub = sub.head(n)
    args = [(i, r, cfg, int(r.batch), False) for cfg in ["QD", "Q75"] for i, r in sub.iterrows()]
    with Pool(10) as pool: res = pool.map(one, args, chunksize=2)
    o = pd.DataFrame(res)
    bad = o[o.status.str.startswith("error")]
    print("first pass", o.groupby("cfg").status.value_counts().to_dict(), flush=True)
    if len(bad):
        args = [(r.i, sub.loc[r.i], r.cfg, int(r.batch), True) for _, r in bad.iterrows()]
        with Pool(10) as pool: rr = pd.DataFrame(pool.map(one, args, chunksize=1))
        o = pd.concat([o[~o.status.str.startswith("error")], rr])
    o = o.merge(sub.drop(columns="batch"), left_on="i", right_index=True)   # original r_f, mu0 for reporting
    o.to_parquet(f"output/uq4_draws{'' if n == 1600 else '_test'}.parquet")
    print("done", o.groupby("cfg").status.value_counts().to_dict(), "mean sec", o.sec.mean())
