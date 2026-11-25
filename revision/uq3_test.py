import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "uq"))
from uq2 import setup
from centred import gap
from centred2 import gap2, solve_c2
d = pd.read_parquet("output/uq2_draws.parquet")
sub = pd.read_parquet(Path(__file__).resolve().parents[1] / "uq/output/uq0_draws.parquet").sample(400, random_state=23)
# agreement on ok draws
for cfg in ["QD"]:
    ok = d[(d.cfg == cfg) & (d.status == "ok")].head(4)
    for _, r in ok.iterrows():
        p, s, sig = setup(sub.loc[r.i], cfg, r.i)
        g0 = gap2(p, 0.0, "F", 0.0); g2 = gap2(p, 0.0, "F", 0.2); k = -g0 / ((g2 - g0) / 0.2)
        print("ok draw", r.i, "old kF", round(r.kF, 4), "new", round(k, 4))
    bad = d[(d.cfg == cfg) & (d.status == "no price")].head(8)
    for _, r in bad.iterrows():
        p, s, sig = setup(sub.loc[r.i], cfg, r.i)
        try:
            g0 = gap2(p, 0.0, "F", 0.0); g2 = gap2(p, 0.0, "F", 0.2); k = -g0 / ((g2 - g0) / 0.2)
            g0u = gap2(p, 0.0, "U", 0.0); g2u = gap2(p, 0.0, "U", 0.2); ku = -g0u / ((g2u - g0u) / 0.2)
            print("was no-price", r.i, "kF", round(k, 4), "kU", round(ku, 4), "psi", round(r.psi, 2))
        except Exception as e:
            print("was no-price", r.i, "still:", str(e)[:70])
