"""Red team v5: pooled implied gap without the dYdX event (buyback-and-stake is not a clean burn)."""
import pandas as pd, numpy as np
d = pd.read_csv("../empirical/output/e2_returns.csv")
for dose in ["dose_high", "dose_low"]:
    for nm, g in [("all", d), ("no dYdX", d[d.event != "dydx-313"])]:
        g = g[g.dose == dose]; w = 1 / g.se**2; m = (w * g.G).sum() / w.sum(); se = 1 / np.sqrt(w.sum())
        print(f"{dose} {nm}: G {m:.1f} se {se:.1f} 95% [{m-1.96*se:.1f}, {m+1.96*se:.1f}]")
