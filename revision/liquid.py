"""Review round 4, item 29: liquid staking. A share phi of stake keeps its transaction convenience, so
n(Theta) = 1 - (1 - phi) Theta; the lock-up cost is unchanged. By Proposition 1 (general form),
kstar = c + L' - alpha D n'/n = r + Gamma/(1-Theta) + alpha D (1-phi)/(1-(1-phi)Theta).
Evaluated on the 19,109 priced prior draws (uq0)."""
import numpy as np, pandas as pd
d = pd.read_parquet("uq/output/uq0_draws.parquet")
phis = np.linspace(0, 1, 21); rows = []
for ph in phis:
    k = d.r + d.Gamma / (1 - d.Theta) + d.alpha * d.D * (1 - ph) / (1 - (1 - ph) * d.Theta)
    x = k / d.r
    rows.append(dict(phi=ph, q05=x.quantile(.05), q50=x.median(), q95=x.quantile(.95),
                     net_share=np.median((d.alpha * d.D * (1 - ph) / (1 - (1 - ph) * d.Theta)) / k)))
o = pd.DataFrame(rows); o.to_csv("revision/output/liquid.csv", index=False)
print(o.round(3).to_string(index=False))
chk = (d.r + d.Gamma / (1 - d.Theta) + d.alpha * d.D / (1 - d.Theta) - d.kstar_cf).abs().max()
print("check vs kstar_cf at phi=0:", chk)
