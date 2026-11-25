import numpy as np, pandas as pd, prop2, uq2
from multiprocessing import Pool
from centred import fee
def f(a):
    i, row, cfg = a
    p, s, sig = uq2.setup(row, cfg, i); P = prop2.pieces(p); z = P["z"]; i0 = int(np.argmin(np.abs(z)))
    _, A = prop2.solve_lin(P, P["dth"], np.zeros(2), sig); m = prop2.weights(P, A, i0, sig)
    wE = m * P["q"] * P["dth"]; wL = wE * P["F"]
    neg = lambda w: float(np.abs(w[P["q"] < 0]).sum() / np.abs(w).sum())
    return dict(i=i, cfg=cfg, negE=neg(wE), negL=neg(wL))
if __name__ == "__main__":
    o = pd.read_parquet("output/uq2_draws.parquet"); o = o[(o.status == "ok") & (o.qmin < 0)]
    d = pd.read_parquet("../uq/output/uq0_draws.parquet")
    args = [(r.i, d.loc[r.i], r.cfg) for r in o.itertuples()]
    with Pool(10) as pool: res = pd.DataFrame(pool.map(f, args))
    print(res.groupby("cfg")[["negE", "negL"]].describe().round(4).T)
    for c, g in res.groupby("cfg"): print(c, "draws with >1% of weight where q<0 (E or L):", int(((g.negE > .01) | (g.negL > .01)).sum()), ">10%:", int(((g.negE > .1) | (g.negL > .1)).sum()))
    res.to_csv("output/uq2_qcheck.csv", index=False)
